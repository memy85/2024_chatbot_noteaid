
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from peft import get_peft_model, PeftModel, PeftConfig, LoraConfig, TaskType
import argparse
import pickle
import random
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

MAX_INPUT_LENGTH = 60000

def load_model_and_tokenizer(MODEL_PATH) :
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_datasets() :

    data = load_dataset("json", data_files={"train" : DATA_PATH.joinpath("train_conversation.jsonl").as_posix(),
                                     "test" : DATA_PATH.joinpath("test_conversation.jsonl").as_posix()})
    return data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--trial_number", type=int, default=0)
    parser.add_argument("--resume", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    return args


def main() :
    args = arguments()
    model_type = args.model
    trial_number = args.trial_number
    resume_checkpoint = args.resume

    MODEL_PATH = config.model_path(model_type)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    print("----------------------   ----------------------------------------------------------  ------------------------------", file=sys.stderr)
    print("----------------------   ------------------------ Logging -------------------------  ------------------------------", file=sys.stderr)
    print("----------------------   ----------------------------------------------------------  ------------------------------", file=sys.stderr)
    print(f"MODEL is : {model_type}\n", file=sys.stderr)
    print(f"MODEL MAX LENGTH is : {tokenizer.model_max_length}\n", file=sys.stderr)
    print(f"PRESET MAX LENGTH is : {MAX_INPUT_LENGTH}\n", file=sys.stderr)


    dataset = load_datasets()

    # first format the chat as the chat template
    dataset = dataset.map(lambda x : {"formatted_chat": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})

    max_length = 0
    for i in dataset['train']['formatted_chat'] :  
        if len(i) > max_length :
            max_length = len(i)

    for i in dataset['test']['formatted_chat'] :  
        if len(i) > max_length :
            max_length = len(i)

    print(f"max length : {max_length}", file=sys.stderr)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    def formatting_func(example) :
        return example['formatted_chat']

    # We do Lora Finetuning
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    
    max_steps = (14231 // (1 * 5))
    print(f"max steps are : {max_steps}", file=sys.stderr)

    training_args = SFTConfig(
        max_seq_length=MAX_INPUT_LENGTH,
        output_dir=f"./model/{model_type}/chatbot{trial_number}/",
        logging_dir=f"./logs/{model_type}/train_{trial_number}/",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=10,
        eval_steps=20,
        logging_steps=1,
        num_train_epochs=100,
        # max_steps=max_steps,
        save_total_limit=10,
        weight_decay=0.01,
        gradient_accumulation_steps=5,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        report_to='tensorboard',
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        args = training_args,
        formatting_func=formatting_func,
    )

    # start training
    print("--------------------------------------- Start Training ----------------------------------------", file=sys.stderr)
    if resume_checkpoint :
        trainer.train(resume_from_checkpoint=True)
    else :
        trainer.train(resume_from_checkpoint=False)

    # save model
    trainer.save_state()
    trainer.save_model(output_dir=f"./model/{model_type}/chatbot{trial_number}")
    tokenizer.save_pretrained(output_dir=f"./model/{model_type}/chatbot{trial_number}")

if __name__ == "__main__" :
    main()



