export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# vllm serve --model /home/htran/generation/med_preferences/AgentClinic/models/Llama3.2-3B-SFT \
# --model /home/htran/generation/med_preferences/AgentClinic/models/Llama3.2-3B-SFT \
python3 -m vllm.entrypoints.openai.api_server \
--model /home/wjang/2024_chatbot_noteaid/model/gguf/llama3.2-3B-lora \
--gpu-memory-utilization 0.5 --tensor-parallel-size 1 \
--served-model-name llama32_3b \
--max-model-len 16000 \
--tokenizer-pool-size 128 --swap-space 8 --block-size 32 \
--port 4999
# --tokenizer-pool-size 128 --swap-space 8 --block-size 32 \

