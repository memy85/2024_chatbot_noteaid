#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# deploy model 
python3 -m vllm.entrypoints.openai.api_server \
--model /home/wjang/2024_chatbot_noteaid/model/gguf/llama3.2-3B/chatbot14 \
--gpu-memory-utilization 0.8 --tensor-parallel-size 1 \
--served-model-name llama32_3b \
--max-model-len 16000 \
--tokenizer-pool-size 128 --swap-space 8 --block-size 32 \
--port 3999
