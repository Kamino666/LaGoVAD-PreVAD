export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=`pwd`:$PYTHONPATH
export HTTPS_PROXY=http://127.0.0.1:7890

python src/end2end_inference.py \
    --config "ckpts/config.yaml" \
    --ckpt "ckpts/best.ckpt" \
    -vp assets/Bullet.in.the.Head.1990__#00-17-20_00-18-55_label_B1-0-0.mp4