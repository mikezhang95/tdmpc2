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

# checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/latent60/models/final.pt

# offline
# checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-dummy_enc-rho1.0/models/final.pt # 556.1 dummy_enc-rho1.0
# checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-sep_enc-rho1.0/models/final.pt # 707.9 sep_enc-rho1.0
# checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-global_enc-rho1.0/models/final.pt # 558.6/205.0 global_enc-rho1.0
checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-global_enc-rho1.0-rand_a/models/final.pt # 685.1/384.9 global_enc-rho1.0
checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-global_enc-rho1.0-rand_a_std0.2/models/final.pt # 665.7/343.8 
checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-global_enc-rho1.0-rand_a_s20/models/final.pt # 689.0/387.6
checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-global_enc-rho1.0-rand_a_s40_std2.0/models/final.pt # 695.8/560.3
checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-global_enc-rho1.0-rand_a-prb/models/final.pt # 667.3/202.9
checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/offline-global_enc-rho1.0-rand_a_s40_std2.0-prb/models/final.pt # 719.0/456.3
checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/latent60/models/final.pt

# online version
# checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/online-dummy_enc-rho1.0/models/final.pt # 773.7/728.8 dummy_enc-rho1.0
# checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/online-sep_enc-rho1.0/models/final.pt # 756.1/679.0 sep_enc-rho1.0
# checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/online-global_enc-rho1.0/models/final.pt # 773.4/758.4 global_enc-rho1.0
# checkpoint=/home/yzhang/tdmpc2/tdmpc2/logs/walker-run/1/online-global_enc-rho1.0-std2/models/final.pt # 747.9/744.6 global_enc-rho1.0

python -u evaluate.py task=walker-run checkpoint=$checkpoint eval_episodes=10 save_video=True exp_name=tmp
