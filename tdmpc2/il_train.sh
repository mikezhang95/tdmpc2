#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 10000 # memory pool for each core (50GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 10 # number of cores
#SBATCH -o log/%x.%A.%a.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log/%x.%A.%a.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J tdmpc2 # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
seed=$((SLURM_ARRAY_TASK_ID+2025))
echo "Seed $seed"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# 1. online training: see online_train.sh:w

# 2. imitation learning
task_name=walker-run
# exp_name=il-lalqr-dyn_comp_fix-cost_diag_fix
exp_name=il-lalqr-baseline

python -u train.py  --config-path=./configs --config-name=tdmpc \
                    task=${task_name} \
                    exp_name=${exp_name} \
                    data_dir=${PWD}/logs/${task_name}/1/online-tdmpc/buffer.pt \
                    checkpoint=${PWD}/logs/${task_name}/1/online-tdmpc/models/final.pt \
                    steps=500000 \
                    enable_wandb=true \
                    compile=true \
                    wandb_project=lalqr \
                    student_cfg=lalqr \
                    # student_cfg.dynamic_structure=companion_fixed \
                    # student_cfg.cost_structure=diag_fixed \

