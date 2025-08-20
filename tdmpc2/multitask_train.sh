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

# offline training
task_name=mt30 # 6

num_agents=6
exp_name=il-fac_N6

python -u train.py  --config-path=./configs --config-name=tdmpc2 \
                    task=${task_name} \
                    exp_name=${exp_name} \
                    data_dir=${PWD}/logs/${task_name} \
                    checkpoint=${PWD}/logs/${task_name}/1/offline-tdmpc2/models/final.pt \
                    steps=1000000 \
                    eval_episodes=1 \
                    eval_freq=100000 \
                    data_ratio=0.5 \
                    lr=0.0 \
                    compile=true \
                    student_cfg=fac_tdmpc \
                    student_cfg.num_agents=${num_agents} \