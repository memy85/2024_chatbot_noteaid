#!/bin/bash
# ------- HIEU's chatbot codes
#
start_chatbot_server(){

	## start chatbot server
	# source activate /home/htran/miniconda3/envs/handbook
	cd /home/htran/generation/med_preferences/AgentClinic
	sh scripts/deploy_llama32_3b_ehr_sft.sh

}


## run demo
run_demo() {
	# source activate /home/htran/miniconda3/envs/genenv
	cd /home/htran/generation/med_preferences/AgentClinic
	python3 noteaid_agent.py
}


"$@"
