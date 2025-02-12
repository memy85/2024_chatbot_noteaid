import torch
import openai
import os
import argparse
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig, AutoModelForCausalLMWithValueHead

from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")
client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)

PHYSICIAN_MODEL_PATH = PROJECT_PATH.joinpath("model/llama3.2-1B/chatbot4/checkpoint-3330")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix())
physician_model = AutoModelForCausalLMWithValueHead.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix(), device_map="auto")

# Load reference model (unchanged baseline)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix(), device_map="auto")

# DPO Training Configuration
dpo_config = DPOConfig(
    beta=0.1,  # Regularization parameter
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1
)

# Initialize DPO Trainer
dpo_trainer = DPOTrainer(
    config=dpo_config,
    model=physician_model,
    ref_model=ref_model,
    tokenizer=tokenizer
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
        physician_inputs = tokenizer(physician_input, return_tensors="pt").to(dpo_trainer.model.device)
        
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

# Function to structure DPO preference data
def get_dpo_preference_data(conversation_text, gold_questions=None):
    """
    Extracts preference-based training data for DPO.
    Returns pairs of (preferred_response, rejected_response).
    """
    # Split conversation history
    split_convo = conversation_text.split("[PHYSICIAN]")

    if len(split_convo) < 3:
        return None  # Not enough turns

    last_physician_response = split_convo[-1].split("[PATIENT]")[0].strip()
    previous_physician_response = split_convo[-2].split("[PATIENT]")[0].strip()

    # Determine which response is better
    reward_last = sum(1 for q in gold_questions if q.lower() in last_physician_response.lower())
    reward_previous = sum(1 for q in gold_questions if q.lower() in previous_physician_response.lower())

    if reward_last > reward_previous:
        return previous_physician_response, last_physician_response  # Last response is better
    else:
        return last_physician_response, previous_physician_response  # Previous response is better

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llama3.2-1B")
    args = parser.parse_args()
    return args 

def main(): 
    args = arguments()
    MODEL_TYPE = args.model_type

    # Training Data
    discharge_note = """<YOUR DISCHARGE NOTE CONTENT HERE>"""
    gold_questions = [
        "What medications do I need to take?",
        "When should I schedule my follow-up appointment?",
        "Are there any dietary restrictions?"
    ]

    num_episodes = 5
    num_epochs = 2

    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1} ===")

        dpo_data = []  # Store preference data

        for episode in range(num_episodes):
            # Generate a conversation with the current policy
            conversation_text = run_conversation(discharge_note, max_turns=3)

            # Get preference data
            preference_pair = get_dpo_preference_data(conversation_text, gold_questions)

            if preference_pair:
                dpo_data.append(preference_pair)
            
            print(f"Episode: {episode+1}, Collected: {len(dpo_data)} pairs")

        # DPO Training Step
        if dpo_data:
            preferred, rejected = zip(*dpo_data)

            preferred_tensors = tokenizer(list(preferred), return_tensors="pt", padding=True, truncation=True)
            rejected_tensors = tokenizer(list(rejected), return_tensors="pt", padding=True, truncation=True)

            dpo_trainer.step(
                preferred_tensors["input_ids"],
                rejected_tensors["input_ids"]
            )

        # Save updated model
        save_path = PROJECT_PATH.joinpath(f"model/dpo_physician_epoch_{epoch+1}")
        dpo_trainer.model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    print("DPO Training Complete!")

if __name__ == "__main__":
    main()

