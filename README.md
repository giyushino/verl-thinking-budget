# verl-thinking-budget
GRPO with a thinking budget, uses verl as a base


## Set Up
```sh 
git clone git@github.com:giyushino/verl-thinking-budget.git
cd verl-thinking-budget
conda create -n thinking python==3.12
conda activate thinking
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install -e . --no-deps
cd ..
```
## Important files to look at
    - verl-thinking-budget/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
