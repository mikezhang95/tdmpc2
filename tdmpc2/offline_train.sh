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

# # 1. online training, get expert model and replay buffer
# task_name=walker-run
# python -u train.py task=${task_name} \
#                 exp_name=online-tdmpc \
#                 num_envs=4 \
#                 steps_per_update=4 \
#                 compile=True



# 2. imitation learning, train fac model
num_agents=6
exp_name=il-fac_N${num_agents}

task_name=walker-run # 6
# task_name=fish-swim # 5
# task_name=reacher-three-easy # 3
# task_name=hopper-stand # 4
# task_name=quadruped-walk # 12 num_noises=20

# task_name=walker-walk # 6
# task_name=hopper-hop # 4
# task_name=hopper-hop-backwards # 4

python -u train.py  --config-path=./configs --config-name=il_fac \
                    task=${task_name} \
                    num_agents=${num_agents} \
                    exp_name=${exp_name} \
                    data_dir=${PWD}/logs/${task_name}/1/online-tdmpc/buffer.pt \
                    checkpoint=${PWD}/logs/${task_name}/1/online-tdmpc/models/final.pt \
                    steps=500000 \
                    compile=True \
		    seed=1 &

python -u train.py  --config-path=./configs --config-name=il_fac \
                    task=${task_name} \
                    num_agents=${num_agents} \
                    exp_name=${exp_name} \
                    data_dir=${PWD}/logs/${task_name}/1/online-tdmpc/buffer.pt \
                    checkpoint=${PWD}/logs/${task_name}/1/online-tdmpc/models/final.pt \
                    steps=500000 \
                    compile=True \
		    seed=2 &

python -u train.py  --config-path=./configs --config-name=il_fac \
                    task=${task_name} \
                    num_agents=${num_agents} \
                    exp_name=${exp_name} \
                    data_dir=${PWD}/logs/${task_name}/1/online-tdmpc/buffer.pt \
                    checkpoint=${PWD}/logs/${task_name}/1/online-tdmpc/models/final.pt \
                    steps=500000 \
                    compile=True \
		    seed=3 &

