#!/bin/bash

# This is command to export for RUN
# export RUN='docker run -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 -e TF_FORCE_GPU_ALLOW_GROWTH=true --shm-size=2g --runtime=nvidia --gpus all --rm -it --privileged -v /home/mohitkhatwani/meta-ckpt:/app maxtext_base_image bash -c'

# gsutil cp command for getting meta-ckpt
# gsutil cp -r gs://maxtext-llama/llama-3-8b/meta-ckpt .

$RUN 'torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir /app --tokenizer_path /app/tokenizer.model --max_seq_len 128 --max_batch_size 4' 
