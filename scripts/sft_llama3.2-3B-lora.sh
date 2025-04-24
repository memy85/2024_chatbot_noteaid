#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2
export MODEL="llama3.2-3B"
export TRIAL_NUMBER=11
export PROJECT_DIR="/home/wjang/2024_chatbot_noteaid"
export MODEL_DIR="/home/experiment_data/wjang/2024_chatbot_noteaid/model/chatbot$TRIAL_NUMBER/"

cd $PROJECT_DIR
python3 src/autoeval_framework_p2_finetune.py --model $MODEL --trial_number $TRIAL_NUMBER --resume True --lora True


