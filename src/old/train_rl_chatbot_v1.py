import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

PHYSICIAN_MODEL_PATH = PROJECT_PATH.joinpath("model/llama3.2-1B/chatbot4/checkpoint-3330")
PATIENT_MODEL_PATH = # this should be the OpenAI chatbot

# ------------------------------------------------------------------
# 1. Load Models and Tokenizer
# ------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix())

# We create a 'policy model' with a value head for PPO.
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    PHYSICIAN_MODEL_PATH.as_posix()
)
# Typically, we also keep a separate reference model for PPO.
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    PHYSICIAN_MODEL_PATH.as_posix()
)
# For the patient, you can simply load the same or a different model
# or rule-based. For demonstration, let's just load the plain LM.
patient_model = AutoModelForCausalLM.from_pretrained(patient_model_name)

# ------------------------------------------------------------------
# 2. Set up the PPO Configuration
# ------------------------------------------------------------------
ppo_config = PPOConfig(
    model_name=PHYSICIAN_MODEL_PATH.as_posix(),
    learning_rate=1.41e-5,
    batch_size=1,            # small for demonstration
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    optimize_cuda_graph=False
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

# ------------------------------------------------------------------
# 3. Define Your Reward Function
# ------------------------------------------------------------------
def compute_reward(conversation_text, gold_questions=None):
    """
    This is a simplified example: we check how many 'questions'
    the patient asked that match a set of 'gold' or desired questions.
    
    - conversation_text: The entire conversation transcript (physician + patient).
    - gold_questions   : A list of question strings we want the patient to ask.

    Return a scalar reward.
    """
    # For demonstration, let's say we just count question marks or
    # count how many gold questions appear in the text.
    reward = 0.0
    
    # Very naive approach: check occurrences of each gold question:
    if gold_questions:
        for gq in gold_questions:
            if gq.lower() in conversation_text.lower():
                reward += 1.0
    
    # Or check how many "?" are present (simple question indicator)
    # reward += conversation_text.count("?") * 0.1

    return reward

# ------------------------------------------------------------------
# 4. Simulate a Conversation (Rollout)
# ------------------------------------------------------------------
def run_conversation(physician_model, patient_model, discharge_note, tokenizer, max_turns=3):
    """
    A minimal example of a "conversation loop" between:
      - physician_model (the one we are training with PPO)
      - patient_model   (a possibly different or rule-based agent)
    
    We return the final conversation text for reward calculation.
    """
    
    conversation_history = []
    
    # System message or context for the physician
    # Possibly you pass in your system instructions + the discharge note
    system_message = f"[SYSTEM]\nYou are a helpful physician. Here is the discharge note:\n{discharge_note}\n"
    conversation_history.append(system_message)
    
    # Start the conversation
    patient_prompt = "[PATIENT] Hi doctor, I have some questions about my discharge note.\n"
    conversation_history.append(patient_prompt)
    
    for turn in range(max_turns):
        # 4.1 Physician's turn
        physician_input = "".join(conversation_history)
        physician_inputs = tokenizer(physician_input, return_tensors="pt")
        # Generate response from the physician (policy model)
        with torch.no_grad():
            physician_output_ids = physician_model.generate(
                **physician_inputs,
                max_new_tokens=50,
                do_sample=True
            )
        physician_output = tokenizer.decode(physician_output_ids[0], skip_special_tokens=True)
        
        # Extract only the newly generated part
        physician_response = physician_output[len(physician_input):]
        conversation_history.append(f"[PHYSICIAN] {physician_response}\n")
        
        # 4.2 Patient's turn
        patient_input = "".join(conversation_history)
        patient_inputs = tokenizer(patient_input, return_tensors="pt")
        with torch.no_grad():
            patient_output_ids = patient_model.generate(
                **patient_inputs,
                max_new_tokens=50,
                do_sample=True
            )
        patient_output = tokenizer.decode(patient_output_ids[0], skip_special_tokens=True)
        
        # Extract only newly generated part
        patient_response = patient_output[len(patient_input):]
        conversation_history.append(f"[PATIENT] {patient_response}\n")
        
    # Return the entire conversation
    full_conversation = "".join(conversation_history)
    return full_conversation

# ------------------------------------------------------------------
# 5. PPO Training Loop
# ------------------------------------------------------------------
discharge_note = """<YOUR DISCHARGE NOTE CONTENT HERE>"""
gold_questions = [
    "What medications do I need to take?",
    "When should I schedule my follow-up appointment?",
    "Are there any dietary restrictions?"
]

num_episodes = 5  # how many conversations per epoch
num_epochs = 2

for epoch in range(num_epochs):
    print(f"=== Epoch {epoch+1} ===")
    
    for episode in range(num_episodes):
        # 5.1 Generate a conversation with the current policy
        conversation_text = run_conversation(
            physician_model=ppo_trainer.model,
            patient_model=patient_model,
            discharge_note=discharge_note,
            tokenizer=tokenizer,
            max_turns=3
        )
        
        # 5.2 Compute the reward
        reward = compute_reward(conversation_text, gold_questions=gold_questions)
        
        # 5.3 PPO requires tokenized 'query' and 'response' portions
        #     In this toy example, we consider the last physician response as 'response'
        
        # Let's extract the last physician response from conversation_text:
        # (A naive approach that just splits on `[PHYSICIAN]`)
        splitted = conversation_text.split("[PHYSICIAN]")
        if len(splitted) > 1:
            last_physician_response = splitted[-1].split("[PATIENT]")[0].strip()
        else:
            last_physician_response = ""
        
        # We also need the "query" (the prompt that preceded the last physician output).
        # For simplicity, let's just take everything up to the last physician response as the "query".
        query = conversation_text.replace(f"[PHYSICIAN] {last_physician_response}", "")
        
        # 5.4 Tokenize query & response
        query_tensors = tokenizer(query, return_tensors="pt")
        response_tensors = tokenizer(last_physician_response, return_tensors="pt")

        # 5.5 Run a single PPO update
        #     We pass the query, the response, and the reward to ppo_trainer.step
        ppo_trainer.step(
            query_tensors["input_ids"],
            response_tensors["input_ids"],
            torch.tensor([reward])
        )
        
        print(f"Episode: {episode+1}, Reward: {reward}")
    
    # At the end of each epoch, you could save your updated model
    ppo_trainer.model.save_pretrained(f"ppo_physician_epoch_{epoch+1}")
    tokenizer.save_pretrained(f"ppo_physician_epoch_{epoch+1}")

print("PPO training complete!")

