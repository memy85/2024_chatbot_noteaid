#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# deploy model 
python3 -m vllm.entrypoints.openai.api_server \
--model /home/wjang/2024_chatbot_noteaid/model/gguf/llama3.2-3B/chatbot1 \
--gpu-memory-utilization 0.5 --tensor-parallel-size 1 \
--served-model-name llama32_3b \
--max-model-len 16000 \
--tokenizer-pool-size 128 --swap-space 8 --block-size 32 \
--port 2999

# vllm serve /home/wjang/2024_chatbot_noteaid/model/gguf/llama3.2-3B/chatbot1 \
# --gpu-memory-utilization 0.5 --tensor-parallel-size 1 \
# --served-model-name llama32_3b \
# --max-model-len 16000 \
# --tokenizer-pool-size 128 --swap-space 8 --block-size 32 \
# --port 2999
	

# --dtype float \
# --model /data/data_user/public_models/Llama-3.2/Llama-3.2-3B-Instruct/ \

# Start simulation using gold discharge note
# python3 src/c2_simulate_conversation.py \
# 	--doctor_llm llama3.2-3B \
# 	--patient_llm gpt-4o-mini 


# Start evaluation 


# Finish session 
# kill $MODEL_PID
