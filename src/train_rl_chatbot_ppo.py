import torch
import torch.nn as nn
import pandas as pd
import openai
from openai import OpenAI
from datasets import Dataset
import os
import argparse
from functools import partial
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, AutoModelForSequenceClassification
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
from trl import (PPOTrainer, 
                 PPOConfig,
                 ScriptArguments,
                 AutoModelForCausalLMWithValueHead, 
                 RewardConfig,
                 RewardTrainer,
                 ModelConfig,
                 get_peft_config)

from utils import *

config = load_config()
logger = config.load_logger("rl_ppo")
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MODEL_PATH = PROJECT_PATH.joinpath("model")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")
client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)

PHYSICIAN_MODEL_PATH = PROJECT_PATH.joinpath("model/llama3.2-1B/chatbot4/checkpoint-3330")
PPO_PHYSICIAN_MODEL_PATH = PROJECT_PATH.joinpath("model/ppo-physician")


# Define a function to call GPT-4o Mini for Patient Agent
def get_patient_response(conversation, max_token=100):
    """
    Uses OpenAI GPT-4o Mini as the Patient Agent to generate responses.
    conversation : list of dictionaries which are the dialogue between the two agents.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        temperature=0.7,
        max_tokens=max_token
    )
    # print(response)

    print(response.choices[0].message.content)
    return response.choices[0].message.content


# Define a function to simulate a conversation
def run_conversation(policy, tokenizer, discharge_note, max_turns=3, device="cpu"):

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
    educator_system_prompt = {"role":"system","content": system_message.format(discharge_note=discharge_note)}
    conversation_history.append(educator_system_prompt)
    
    # Patient starts the conversation
    # Choose initiating conversation : general "Hi", question ~?. 
    patient_system_prompt = {"role": "system", "content": config.load_prompts("rl_patient.txt")}
    patient_prompt = {"role": "user", "content" : "Hi, I have some questions about my discharge note.\n"}
    conversation_history.append(patient_prompt)

    for turn in range(max_turns):
        # Physician Agent (LLaMA 3.2-1B) generates response
        physician_inputs = tokenizer.apply_chat_template([educator_system_prompt] + conversation_history[1:], 
                                                         tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        # print(physician_inputs)
        
        with torch.no_grad():
            physician_output_ids = policy.generate(
                physician_inputs,
                **generation_config
            )
        
        arr_output = physician_output_ids.detach().cpu().numpy()
        start_of_generate_index = physician_inputs.shape[1]
        physician_response = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]

        # physician_response = tokenizer.decode(physician_output_ids[0], skip_special_tokens=True)
        # new_physician_response = physician_response[len(physician_inputs):]  # Extract newly generated response
        print(f"physician: {physician_response}")
        conversation_history.append({"role": "assistant", "content" : physician_response + "\n"})

        # Patient Agent (GPT-4o Mini) responds
        patient_response = get_patient_response([patient_system_prompt] + conversation_history[1:]) # Because we use different system prompts for each agents
        print(f"patient: {patient_response}")
        conversation_history.append({"role" : "user", "content" : patient_response})
        # change system message for physician model
        conversation_history[0] = educator_system_prompt

    # Return full conversation transcript
    return conversation_history


# Define a function to compute reward based on multiple-choice test results
def compute_reward(patient_answers, gold_answers):
    """
    Computes a reward based on the correctness of the physician model's performance on a multiple-choice test.
    Each correct answer adds 1 point to the total score.
    """
    reward = sum(1 for pa, ga in zip(patient_answers, gold_answers) if pa == ga)
    return reward / len(gold_answers)  # Normalize reward

                                    # Function to simulate the conversation and test the physician model
def evaluate_student_model(discharge_note, questions, conversation_history):
    """
    Runs a simulated conversation and then presents multiple-choice questions to the patient model.
    The reward is based on the number of correct answers.
    """
    # change system prompt in conversation history for evaluation
    evaluation_stystem_message = {"role":"system","content": "You will be given a conversation history between you(user) and the physician about your discharge note. Based on the conversation history, you will be queried to answer a question with 3 choices: a,b,c. Among the given choices choose the one that is the most appropriate. For example, if you think choice a is the answer, output a"}
    conversation_history[0] = evaluation_stystem_message
    
    gold_answers = [q['answer'] for q in questions]
    q_list = [q['question'] for q in questions]
    choice_list = [q['choices'] for q in questions]
    
    def reformat_choices(choices) :
        text = ""
        for k, v in choices.items() : 
            text += f"{k}) {v}\n"
        return text
    choice_list = [reformat_choices(c) for c in choice_list]

    # Ask multiple-choice questions to the patient model
    patient_answers = []
    for question, choice in zip(q_list, choice_list):
        question_prompt = f"{question}\nChoices:\n{choice}\n Answer:"
        response = get_patient_response(conversation_history + [{"role": "user", "content": question_prompt}], max_token=2)
        patient_answers.append(response.strip().lower())
    
    # Compute reward based on the test results
    reward = compute_reward(patient_answers, gold_answers)
    return reward, patient_answers

class CustomRewardModel(nn.Module) :

    def __init__(self, reward_fn, tokenizer) :
        super().__init__()
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

    def forward(self, query_ids, response_ids, sample_data) :
        # Convert token IDs to text (you may need to adjust based on your formatting)
        conversation_context = self.tokenizer.decode(query_ids[0], skip_special_tokens=True)
        response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Combine the conversation context and response
        full_conversation = conversation_context + "\n" + response_text
        
        # Compute reward using your custom reward function
        reward_value = self.reward_fn(full_conversation)
        
        # Return a tensor (make sure it's on the same device as the inputs)
        return torch.tensor([reward_value], dtype=torch.float32, device=query_ids.device)



def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llama3.2-1B")
    args = parser.parse_args()
    return args 


def main() : 

    args = arguments()
    MODEL_TYPE = args.model_type
    # Load dataset
    rl_dataset = pd.read_pickle(DATA_PATH.joinpath("rl_dataset.pkl"))
    dataset = Dataset.from_pandas(rl_dataset)
    # dataset['question_rl'][0]

    # Load your trained Physician model (LLaMA 3.2-1B fine-tuned)
    tokenizer = AutoTokenizer.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix())
    policy_model = AutoModelForCausalLM.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix(), device_map="auto")
    policy = PeftModelForCausalLM.from_pretrained(policy_model, PHYSICIAN_MODEL_PATH.as_posix())
    logger.debug("Loaded physician model")

    # physician_model = AutoModelForCausalLMWithValueHead.from_pretrained(model, device_map="auto")
    logger.debug("Loaded with valuehead physician model")

    # Keep a reference model for PPO (unchanged model for stability)
    # ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(PHYSICIAN_MODEL_PATH.as_posix(), device_map="auto")
    ref_policy = PeftModelForCausalLM.from_pretrained(ref_model, PHYSICIAN_MODEL_PATH.as_posix())
    logger.debug("Loaded with reference physician model")

    # PPO Training Configuration
    training_args = PPOConfig(
        output_dir=PPO_PHYSICIAN_MODEL_PATH,
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=policy,
        value_model=None,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    device = ppo_trainer.model.device

    # PPO Training Loop
    num_episodes = 5
    num_epochs = 2

    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1} ===")
        for episode in range(num_episodes):
            training_samples = []
            for i, data in enumerate(dataset): 
                discharge_note = data['text']
                questions = data['question_rl']

                # Generate a conversation with the current policy
                conversation_history = run_conversation(policy= policy, tokenizer= tokenizer, 
                                                        discharge_note=discharge_note, max_turns=3, device=device)

                # Compute the reward
                reward, patient_answers = evaluate_student_model(discharge_note, questions, conversation_history)

                # Tokenize the last physician response and use it in PPO training
                full_conversation = "\n".join([entry["content"] for entry in conversation_history])
                query_tensors = tokenizer(full_conversation, return_tensors="pt").to(device)
                response_tensors = tokenizer(conversation_history[-1]["content"], return_tensors="pt").to(device)

                training_samples.append({
                "input_ids": query_tensors["input_ids"],
                "response_ids": response_tensors["input_ids"],
                "reward": torch.tensor([reward], dtype=torch.float32).to(ppo_trainer.model.device)
                })

                if i == 2 :
                    break


            # custom_dataset = Dataset.from_list(training_samples)
            # ppo_trainer.train_dataset = custom_dataset
            ppo_trainer.train()
                

        # Save updated model
        ppo_trainer.model.save_pretrained(MODEL_PATH.joinpath(f"ppo_physician_epoch_{epoch+1}"))
        tokenizer.save_pretrained(MODEL_PATH.joinpath(f"ppo_physician_epoch_{epoch+1}"))

    print("PPO Training Complete!")

if __name__ == "__main__" :
    main()
