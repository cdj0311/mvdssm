#!/bin/sh
ckpt_dir=./ckpt
user_model_path=./user_model
item_model_path=./item_model

train_data=train_files.txt
eval_data=eval_files.txt
train_steps=100000000
batch_size=256
learning_rate=0.000005
save_steps=10000
embed_size=32

python main.py \
    --train_data=${train_data} \
    --eval_data=${eval_data} \
    --model_dir=${ckpt_dir} \
    --user_model_path=${user_model_path} \
    --item_model_path=${item_model_path} \
    --train_steps=${train_steps} \
    --save_checkpoints_steps=${save_steps} \
    --learning_rate=${learning_rate} \
    --batch_size=${batch_size} \
    --is_eval=False \
    --run_on_cluster=False \
    --train_eval=True \
    --export_user_model=True \
    --export_item_model=True \
    --embed_size=${embed_size} \
    --gpuid=3
