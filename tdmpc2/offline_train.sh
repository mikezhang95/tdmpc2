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

python -u train.py  task=walker-run \
                    num_envs=5 \
                    exp_name=offline2-dummy_enc-cold_tdmpc \
                    data_dir=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/baseline/buffer.pt \
                    steps=600000 \
                    compile=True \
                    helper_checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/latent60 \
                    # helper_checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline_tdmpc2/ \
