
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, PeftModel, PeftConfig, LoraConfig, TaskType
import argparse
import pickle
import random
from utils import *

config = load_config()
filename = os.path.basename(__file__)
logging = config.load_logger(filename)
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
# SAVE_PATH = Path("/data/experiment_data/wjang/2024_chatbot_noteaid")
SAVE_PATH = PROJECT_PATH

device_counts = torch.cuda.device_count()
logging.debug(f"NUM GPU : {device_counts}")

MAX_INPUT_LENGTH = 60000

def load_model_and_tokenizer(MODEL_PATH) :
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def form_dataset() :
    with open(DATA_PATH.joinpath("synthetic_notes_for_sft.pkl"), "rb") as f :
        notes = pickle.load(f)
    
    with open(DATA_PATH.joinpath("synthetic_conversation_10000.pkl"), "rb") as f :
        chat = pickle.load(f)
        # sft_chat = []
        # for i in notes.index :
        #     sft_chat.append(chat[i])
        # chat = sft_chat

    prompt = config.load_prompts("autoeval_stage1.txt")
    # base = "You are a doctor who is having a conversation with a patient to help them understand the key details of their discharge note. You will explain their diagnosis, treatment plan, medications, and follow-up instructions in a clear and supportive manner. Your dialogue will be 1-3 sentences in length, and you should encourage the patient to ask questions if anything is unclear. If you don't have any questions to ask the patient, say <end_of_conversation>."
    # new_rows = []
    # for i, row in notes.iterrows():
    #     #     # q = str(q)
    #     note = row['note']
    #     conversation = chat[i]
    #
    #     # Add <end of conversation> at the end
    #     if conversation[-1]['role'] == 'assistant':
    #         conversation[-1]['content'] += " <end_of_conversation>"
    #     else:
    #         conversation.append({
    #             'role': 'assistant',
    #             'content': 'Thanh you! <end_of_conversation>'
    #         })
    #     # system_prompt = prompt.format(discharge_note=note)
    #     presentation = "\nBelow is the discharge note of the patient: {}. \n".format(note)
    #     system_prompt = base + presentation
    #     new_rows.append({"messages": [{"role": "system", "content": system_prompt}] + conversation})
    #     # import ipdb; ipdb.set_trace()
    # new_rows = Dataset.from_list(new_rows)
    # return new_rows


    new_rows = []
    for i, row in notes.iterrows() :
    #     # q = str(q)
        note = row['note']
        conversation = chat[i]

        # Add <end of conversation> at the end 
        conversation[-1]['content'] += " <end of conversation>"
        system_prompt = prompt.format(discharge_note=note)

        new_rows.append({"messages" : [{"role" : "system", "content" : system_prompt}] + conversation })
    new_rows = Dataset.from_list(new_rows)
    return new_rows

def load_datasets() :
    df = form_dataset()
    # df = Dataset.from_list(df)
    df = df.train_test_split(train_size=0.8, seed=0)
    return df

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
    parser.add_argument("--lora", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--trial_number", type=int, default=0)
    parser.add_argument("--resume", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    return args


def main() :
    args = arguments()
    model_type = args.model
    lora_flag = args.lora
    trial_number = args.trial_number
    resume_checkpoint = args.resume
    
    # model_type = "llama3.2-3B"
    MODEL_PATH = config.model_path(model_type)
    
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    logging.debug("----------------------   ----------------------------------------------------------  ------------------------------")
    logging.debug("----------------------   ------------------------ Logging -------------------------  ------------------------------")
    logging.debug("----------------------   ----------------------------------------------------------  ------------------------------")
    logging.debug(f"MODEL is : {model_type}\n")
    logging.debug(f"MODEL MAX LENGTH is : {tokenizer.model_max_length}\n")
    logging.debug(f"PRESET MAX LENGTH is : {MAX_INPUT_LENGTH}\n")

    dataset = load_datasets()

    # dataset['train']['messages'][0][-1]
    # Apply tokenization and masking to dataset
    # tokenized_datasets = dataset.map(tokenize_and_mask, batched=True)

    # add END OF CONVERSATION to the end of the conversation
    def add_eoc(chat) :
        conversation = chat['messages']
        # conversation[-1]['content'] = conversation[-1]['content'] + "|| END OF CONVERSATION"

        formatted = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        return formatted 

    dataset = dataset.map(lambda x : {"formatted_chat": add_eoc(x)})
    # dataset['train']['formatted_chat'][0]

    max_length = 0
    for i in dataset['train'] :  
        if len(i['messages']) > max_length :
            max_length = len(i['messages'])

    for i in dataset['test'] :  
        if len(i['messages']) > max_length :
            max_length = len(i['messages'])

    logging.debug(f"max length : {max_length}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    def formatting_func(example) :
        return example['formatted_chat']

    # We do Lora Finetuning
    if lora_flag :
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
    
    max_steps = len(dataset['train']) * 5
    logging.debug(f"max steps are : {max_steps}")

    training_args = SFTConfig(
        max_seq_length=MAX_INPUT_LENGTH,
        output_dir=SAVE_PATH.joinpath(f"./model/{model_type}/chatbot{trial_number}"),
        # logging_dir=SAVE_PATH.joinpath(f"./model/{model_type}/chatbot{trial_number}")
        logging_dir=PROJECT_PATH.joinpath(f"./logs/{model_type}/train_{trial_number}/"),
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=50,
        eval_steps=50,
        logging_steps=1,
        num_train_epochs=100,
        # max_steps=max_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        weight_decay=0.01,
        gradient_accumulation_steps=5,
        gradient_checkpointing=True,
        eval_strategy="steps",
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
        trainer.train(resume_from_checkpoint=True);
    else :
        trainer.train(resume_from_checkpoint=False);

    # save model
    # trainer.save_state()
    trainer.save_model(output_dir=SAVE_PATH.joinpath(f"./model/{model_type}/chatbot{trial_number}"))
    tokenizer.save_pretrained(save_directory=SAVE_PATH.joinpath(f"./model/{model_type}/chatbot{trial_number}"))

if __name__ == "__main__" :
    main()



