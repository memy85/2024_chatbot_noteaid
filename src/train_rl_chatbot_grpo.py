import torch
import openai
import os
import argparse
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, GRPOConfig, GRPOTrainer

from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")
client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)


# Define a function to call GPT-4o Mini for Patient Agent
def get_patient_response(conversation_text):
    """
    Uses OpenAI GPT-4o Mini as the Patient Agent to generate responses.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a patient asking about your discharge note."},
                  {"role": "user", "content": conversation_text}],
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]

# Define a function to simulate a conversation
def run_conversation(discharge_note, max_turns=3):
    """
    Runs a simulated conversation using:
      - LLaMA 3.2-1B (Physician Model)
      - GPT-4o Mini (Patient Model via OpenAI API)
    """

    conversation_history = []
    
    # System message & discharge note
    system_message = f"[SYSTEM]\nYou are a helpful educator in medicine. Here is the discharge note:\n{discharge_note}\n"
    conversation_history.append(system_message)
    
    # Patient starts the conversation
    patient_prompt = "[PATIENT] Hi, I have some questions about my discharge note.\n"
    conversation_history.append(patient_prompt)
    
    for turn in range(max_turns):
        # Physician Agent (LLaMA 3.2-1B) generates response
        physician_input = "".join(conversation_history)
        physician_inputs = tokenizer(physician_input, return_tensors="pt").to(ppo_trainer.model.device)
        
        with torch.no_grad():
            physician_output_ids = physician_model.generate(
                **physician_inputs,
                max_new_tokens=50,
                do_sample=True
            )
        
        physician_response = tokenizer.decode(physician_output_ids[0], skip_special_tokens=True)
        new_physician_response = physician_response[len(physician_input):]  # Extract newly generated response
        conversation_history.append(f"[PHYSICIAN] {new_physician_response}\n")

        # Patient Agent (GPT-4o Mini) responds
        patient_response = get_patient_response("".join(conversation_history))
        conversation_history.append(f"[PATIENT] {patient_response}\n")

    # Return full conversation transcript
    return "".join(conversation_history)

# Reward function to evaluate how many "gold label" questions were asked
def compute_reward(conversation_text, gold_questions=None):
    """
    Computes a reward based on the number of 'gold label' questions found in the conversation.
    """
    reward = 0.0
    
    if gold_questions:
        for gq in gold_questions:
            if gq.lower() in conversation_text.lower():
                reward += 1.0
    
    return reward


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llama3.2-1B")
    args = parser.parse_args()
    return args 


def main() : 
    args = arguments()
    MODEL_TYPE = parser.model_type

    PHYSICIAN_MODEL_PATH = PROJECT_PATH.joinpath("model/llama3.2-1B/chatbot4/checkpoint-3330")

    # Load your trained Physician model (LLaMA 3.2-1B fine-tuned)
    tokenizer = AutoTokenizer.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix())
    physician_model = AutoModelForCausalLMWithValueHead.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix(), device_map="auto")

    # Keep a reference model for PPO (unchanged model for stability)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix(), device_map="auto")

    # OpenAI API Key (Set your OpenAI API Key)
    openai.api_key = os.getenv("OPENAI_API_KEY")  # Or manually: openai.api_key = "your-api-key"

    # GRPO Training Configuration
    grpo_config = GRPOConfig(
        model_name=physician_model_path,
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1
    )

    # Initialize GRPO Trainer
    grpo_trainer = GRPOTrainer(
        config=grpo_config,
        model=physician_model,
        ref_model=ref_model,
        tokenizer=tokenizer
    )


    # GRPO Training Loop
    discharge_note = """<YOUR DISCHARGE NOTE CONTENT HERE>"""
    gold_questions = [ "What medications do I need to take?",
        "When should I schedule my follow-up appointment?",
        "Are there any dietary restrictions?"
    ]

    num_episodes = 5
    num_epochs = 2

    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1} ===")
        
        for episode in range(num_episodes):
            # Generate a conversation with the current policy
            conversation_text = run_conversation(discharge_note, max_turns=3)

            # Compute the reward
            reward = compute_reward(conversation_text, gold_questions=gold_questions)

            # Extract last physician response as 'response'
            splitted = conversation_text.split("[PHYSICIAN]")
            last_physician_response = splitted[-1].split("[PATIENT]")[0].strip() if len(splitted) > 1 else ""
            query = conversation_text.replace(f"[PHYSICIAN] {last_physician_response}", "")

            # Tokenize query & response
            query_tensors = tokenizer(query, return_tensors="pt")
            response_tensors = tokenizer(last_physician_response, return_tensors="pt")

            # Run GRPO Update
            grpo_trainer.step(
                query_tensors["input_ids"],
                response_tensors["input_ids"],
                torch.tensor([reward])
            )
            
            print(f"Episode: {episode+1}, Reward: {reward}")

        # Save updated model
        ppo_trainer.model.save_pretrained(MODEL_PATH.joinpath(f"ppo_physician_epoch_{epoch+1}"))
        tokenizer.save_pretrained(MODEL_PATH.joinpath(f"ppo_physician_epoch_{epoch+1}"))

    print("PPO Training Complete!")

