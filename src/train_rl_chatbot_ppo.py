import torch
import openai
from openai import OpenAI
import os
import argparse
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

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
PPO_PHYSICIAN_MODEL_PATH = PROJECT_PATH.joinpath("model/ppo-physician")

# Load your trained Physician model (LLaMA 3.2-1B fine-tuned)
tokenizer = AutoTokenizer.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix())
model = AutoModelForCausalLM.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix(), device_map="auto")
model = PeftModelForCausalLM.from_pretrained(model, PHYSICIAN_MODEL_PATH.as_posix())

physician_model = AutoModelForCausalLMWithValueHead.from_pretrained(model, device_map="auto")

# Keep a reference model for PPO (unchanged model for stability)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model, device_map="auto")


# PPO Training Configuration
ppo_config = PPOConfig(
    model_name=PPO_PHYSICIAN_MODEL_PATH,
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1
)

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=physician_model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

# Define a function to call GPT-4o Mini for Patient Agent
def get_patient_response(conversation, discharge_note):
    """
    Uses OpenAI GPT-4o Mini as the Patient Agent to generate responses.
    conversation : list of dictionaries which are the dialogue between the two agents.
    """

    system_message = "You are a patient having a conversation about your discharge note. The educator might ask you some questions or you could also ask some questions as well.\nHere is your discharge note : {}\n\n Given the discharge note and conversation history, begin or carry on the conversation.".format(discharge_note)

    messages = [{"role": "system", "content": system_message}]
    messages.extend(conversation)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]

def apply_chat_template(conversation_history) :
    chat = ""
    for conv in conversation_history :
        chat += f"{conv['role']}: {conv['content']}"

    return chat

# Define a function to simulate a conversation
def run_conversation(discharge_note, max_turns=3):

    """
    Runs a simulated conversation using:
      - LLaMA 3.2-1B (Physician Model)
      - GPT-4o Mini (Patient Model via OpenAI API)
    """

    system_message = config.load_prompts("ppo_style1.txt")

    generation_config = {
        "max_length" : 60000,
        "max_new_tokens" : 100,
        "repetition_penalty" : 1.2,
        "top_k" : 50,
        "temperature" : 0.9,
        "early_stopping" : True,
        "num_beams" : 5,
        "eos_token_id" : tokenizer.eos_token_id
    }

    conversation_history = []
    
    # System message & discharge note
    conversation_history.append({"role":"system","content": system_message.format(discharge_note)})
    
    # Patient starts the conversation
    # Choose initiating conversation : general "Hi", question ~?. 
    patient_prompt = {"role": "user", "content" : "Hi, I have some questions about my discharge note.\n"}
    conversation_history.append(patient_prompt)

    
    for turn in range(max_turns):
        # Physician Agent (LLaMA 3.2-1B) generates response
        # physician_input = "".join(conversation_history)

        # tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        chat = apply_chat_template(conversation_history)
        physician_inputs = tokenizer(chat, return_tensors="pt").to(ppo_trainer.model.device)
        
        with torch.no_grad():
            physician_output_ids = physician_model.generate(
                physician_inputs,
                **generation_config
            )
        
        physician_response = tokenizer.decode(physician_output_ids[0], skip_special_tokens=True)
        new_physician_response = physician_response[len(physician_input):]  # Extract newly generated response
        conversation_history.append({"role": "educator", "content" : new_physician_response + "\n"}")

        # Patient Agent (GPT-4o Mini) responds
        patient_response = get_patient_response(conversation_history[1:]) # Because we use different system prompts for each agents
        conversation_history.append({"role" : "user", "content" : patient_response})

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

    # PPO Training Loop
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
            conversation_text = run_conversation(discharge_note, max_turns=20)

            # Compute the reward
            reward = compute_reward(conversation_text, gold_questions=gold_questions)

            # Extract last physician response as 'response'
            splitted = conversation_text.split("[PHYSICIAN]")
            last_physician_response = splitted[-1].split("[PATIENT]")[0].strip() if len(splitted) > 1 else ""
            query = conversation_text.replace(f"[PHYSICIAN] {last_physician_response}", "")

            # Tokenize query & response
            query_tensors = tokenizer(query, return_tensors="pt")
            response_tensors = tokenizer(last_physician_response, return_tensors="pt")

            # Run PPO Update
            ppo_trainer.step(
                query_tensors["input_ids"],
                response_tensors["input_ids"],
                torch.tensor([reward])
            )
            
            print(f"Episode: {episode+1}, Reward: {reward}")

        # Save updated model
        ppo_trainer.model.save_pretrained(MODEL_PATH.joinpath(f"ppo_physician_epoch_{epoch+1}"))
        tokenizer.save_pretrained(MODEL_PATH.joinpath(f"ppo_physician_epoch_{epoch+1}"))

    print("PPO Training Complete!")

