export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

config_name=online_tdmpc2
exp_name=rollout-tdmpc2

task_name=quadruped-run
# task_name=cheetah-run
# task_name=humanoid-run

python -u train.py --config-path=./configs --config-name=${config_name} \
                checkpoint=${PWD}/logs/dmcontrol/${task_name}-1.pt \
                exp_name=${exp_name} \
                task=${task_name} \
                lr=0.0 \
                num_envs=10 \
		enable_wandb=False \
		compile=True \
