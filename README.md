# 2024 Chatbot NoteAid

## Abstract
Patients must possess the knowledge necessary to actively participate in their care. We
present NoteAid-Chatbot, a conversational AI that promotes patient understanding via a novel ‘learning as conversation’ framework, built on a multi-agent large language model (LLM) and reinforcement learning (RL) setup without human-labeled data. NoteAid-Chatbot was
built on a lightweight 3B-parameter LLaMA 3.2 model trained in two stages: initial supervised fine-tuning on conversational data synthetically generated using medical conversation strategies, followed by RL with rewards derived from patient understanding assessments in simulated hospital discharge scenarios. Our evaluation, which includes comprehensive human-aligned assessments and case studies, demonstrates that NoteAid-Chatbot exhibits key emergent behaviors critical for patient education—such as clarity, relevance, and structured dialogue—even though it received no explicit supervision for these attributes. Our
results show that even simple Proximal Policy Optimization (PPO)-based reward modeling can successfully train lightweight, domain-specific chatbots to handle multi-turn interactions, incorporate diverse educational strategies, and meet nuanced communication objectives. Our Turing test demonstrates that NoteAid-Chatbot surpasses non-expert human.

Although our current focus is on healthcare, the framework we present illustrates the feasibility and promise of applying low-cost, PPO-based RL to realistic, open-ended conversational domains—broadening the applicability of RL-based alignment methods.

### Activating environment
- python version : `3.12.5`
1. `$ conda create -n myenv --file package-list.txt`
2. `$ conda activate myenv`

### Contributors
- Won Seok Jang, Hieu Tran 
- Occupation : [UMass Lowell](www.uml.edu)

### Citation

`
@article{jang2025chatbot,
  title={Chatbot To Help Patients Understand Their Health},
  author={Jang, Won Seok and Tran, Hieu and Mistry, Manav and Gandluri, SaiKiran and Zhang, Yifan and Sultana, Sharmin and Kown, Sunjae and Zhang, Yuan and Yao, Zonghai and Yu, Hong},
  journal={arXiv preprint arXiv:2509.05818},
  year={2025}
}
`
