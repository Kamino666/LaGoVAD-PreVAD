export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=`pwd`:$PYTHONPATH
export HTTPS_PROXY=http://127.0.0.1:7890
export MPLBACKEND=agg  # Make sure set this during training
export TOKENIZERS_PARALLELISM=false

# Not tested on multi-gpu
python src/main.py fit -c src/configs/default.yaml --trainer.devices=1
