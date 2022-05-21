set -ex
rlaunch --cpu=4 --gpu=1 --memory=10240 -- python ./scripts/evaluate_burstsr_val.py
