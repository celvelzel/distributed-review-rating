#!/bin/bash
#SBATCH --job-name=3m-bs16-ablation
#SBATCH --partition=h07gpuq1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/3m_bs16_ablation_%j.out
#SBATCH --error=logs/3m_bs16_ablation_%j.err

echo "$(date) Starting 3M BS16 ablation experiment"
cd /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating
python -u code/models/deberta_3m_bs16_ablation.py 2>&1
echo "$(date) Done"
