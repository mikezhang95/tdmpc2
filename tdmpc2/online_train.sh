#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 50000 # memory pool for each core (50GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 20 # number of cores
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


# 1. online tdmpc
config_name=online_tdmpc
exp_name=online-tdmpc

python -u train.py --config-path=./configs --config-name=${config_name} \
                exp_name=${exp_name} \
                task=walker-run \
                num_envs=4 \
                steps_per_update=4 \
                compile=True

# 2. online fac tdmpc 
config_name=online_fac
exp_name=online-fac
num_agents=6

python -u train.py --config-path=./configs --config-name=${config_name} \
                exp_name=${exp_name} \
                num_agents=${num_agents}
                task=walker-run \
                num_envs=4 \
                steps_per_update=4 \
                enable_wandb=False \
#                 compile=True


# 3. online attention tdmpc 
config_name=online_attn
exp_name=online-attn

python -u train.py --config-path=./configs --config-name=${config_name} \
                exp_name=${exp_name} \
                task=walker-run \
                num_envs=4 \
                steps_per_update=4 \
                compile=True

