#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# deploy model 
python3 -m vllm.entrypoints.openai.api_server \
--model /home/htran/med_preferences/AgentClinic/ppo_checkpoints/checkpoint-140/ \
--gpu-memory-utilization 0.5 --tensor-parallel-size 1 \
--served-model-name llama32_3b \
--max-model-len 16000 \
--tokenizer-pool-size 128 --swap-space 8 --block-size 32 \
--port 4999
# MODEL_PID=$!
# --model /home/htran/med_preferences/AgentClinic/grpo_checkpoints/checkpoint-latest \

# Start simulation using gold discharge note
# python3 src/c2_simulate_conversation.py \
# 	--doctor_llm llama-3-3B-instruct \
# 	--patient_ll gpt-4o-mini \
#

# Start evaluation 



# Finish session 
# kill $MODEL_PID
