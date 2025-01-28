CUDA_DEVICE_ORDER := PCI_BUS_ID
CUDA_LAUNCH_BLOCKING := 1
CUDA_VISIBLE_DEVICES := 5
TORCH_EXTENSIONS_DIR :=/home/zhichaoyang/.cache/torch_extensions

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export TORCH_EXTENSIONS_DIR

# get chat outputs

pittsburgh_chatdata :
	python3 src/query_chatgpt.py --dataset pittsburgh --save_name pittsburgh_chat > ./logs/pitts.log 2>&1
mimic_chatdata :
	python3 src/query_chatgpt.py --dataset mimic --save_name mimic_chat > ./logs/mimic.log 2>&1


# Train chatbot

train_llama3.2-3B :
	python3 src/train_chatbot.py --model llama3.2-3B --trial_number 1 > ./logs/train_llama3.2.log 2>&1

train_llama3.2-1B :
	echo devices : $$CUDA_VISIBLE_DEVICES > ./logs/train_llama3.2.log
	echo "python3 src.train_chatbot.py --model llama3.2-1B --trial_number 4 --resume True" >> ./logs/train_llama3.2.log 2>&1

	python3 src/train_chatbot.py --model llama3.2-1B --trial_number 4 --resume True >> ./logs/train_llama3.2.log 2>&1

# demo
test_llama3.2-1B :
	python3 src/simulate_chatbot.py

# measure score
measure_score_llama3.2-1B :
	python3 src/measure_score_bleu_rouge_bert.py --model llama3.2-1B --trial 4 

# llama3.2-1B as baseline
# llama3.2-3B as baseline
measure_score_baseline : 
	python3 src/measure_score_bleu_rouge_bert.py --model llama3.2-3B --baseline True
	python3 -c "from src.utils import * ; send_line_message('finished baseline chatbot evaluation');"

measure_score_gpt : 
	python3 src/measure_score_bleu_rouge_bert.py --openai_flag True
	python3 -c "from src.utils import * ; send_line_message('finished gpt4o-mini chatbot evaluation');"
	

# evaluate dataset
evaluate_dataset :
	python3 src/evaluate_dataset.py --type train
	python3 src/evaluate_dataset.py --type test
	python3 -c "from src.utils import * ; send_line_message('finished evaluating the datasets')"

evaluate_dataset_gpt :
	python3 src/evaluate_dataset.py --type train --openai_flag True
	python3 src/evaluate_dataset.py --type test --openai_flag True
