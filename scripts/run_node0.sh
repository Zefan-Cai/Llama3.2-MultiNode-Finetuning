NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 \

MASTER_ADDR=
MASTER_PORT=

torchrun \
--nproc_per_node 8 \
--nnodes 2 \
--node_rank 0 \
--rdzv_backend c10d \
--rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
--master_addr ${MASTER_ADDR} \
--master_port ${MASTER_PORT} \
./finetune.py \
--model_name_or_path "./models/Llama-3.2-3B-Instruct" \
--data_path "./data/train.json" \
--bf16 True \
--output_dir "./outputs/llama3_8B_lora" \
--num_train_epochs 100 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 5 \
--save_total_limit 1 \
--learning_rate 1e-5 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "none" \
--model_max_length 4096 \
--gradient_checkpointing True \
--lazy_preprocess True \
--deepspeed ./config/ds_config_zero2.json \
--use_lora