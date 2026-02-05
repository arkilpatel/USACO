import os

# Define base SLURM script
base_script = """#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/a/arkil.patel/olmo/slurm_logs/usaco-{model}.out

module load python/3.10 cuda/12.6.0/cudnn openmpi

cd ~/USACO

source .venv/bin/activate

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

python run_usaco.py -m {model} -b vllm-local --serve --streaming --expert-passed
"""

models = [
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    # "HuggingFaceTB/SmolLM3-3B",
    # "allenai/Olmo-3-7B-Think",
    # "mistralai/Ministral-3-3B-Reasoning-2512"
]

# Create scripts directory
scripts_dir = "slurm_scripts"
os.makedirs(scripts_dir, exist_ok=True)

# Loop through step values, generate SLURM scripts, and submit them
for model in models:
    script_content = base_script.format(model=model)
    script_filename = os.path.join(scripts_dir, f"usaco_{model.replace('/', '_')}.sh")
    # Write script to file
    with open(script_filename, "w") as f:
        f.write(script_content)

    # Make script executable
    os.chmod(script_filename, 0o755)

    # Submit the job
    os.system(f"sbatch {script_filename}")

print("All jobs submitted!")