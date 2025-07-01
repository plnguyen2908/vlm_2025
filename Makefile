train_4bit_quantization:
	torchrun --nproc_per_node=4 train.py \
    --model_id google/paligemma2-10b-pt-224 \
    --train_data_path processed_dataset/train \
    --val_data_path processed_dataset/validation \
    --output_dir fintuned_paligemma2_4bit_4gpus \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 10 \
	--quant 4bit

# train_8bit_quantization_ddp:
# 	torchrun --nproc_per_node=4 train.py \
#     --model_id google/paligemma2-10b-pt-224 \
#     --train_data_path processed_dataset/train \
#     --val_data_path processed_dataset/validation \
#     --output_dir fintuned_paligemma2_8bit_4_gpus \
#     --num_train_epochs 8 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 64 \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.03 \
#     --lora_r 16 \
#     --lora_alpha 32 \
#     --lora_dropout 0.05 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --logging_steps 10 \
# 	--quant 8bit

train_8bit_quantization:
	python train.py \
    --model_id google/paligemma2-10b-pt-224 \
    --train_data_path processed_dataset/train \
    --val_data_path processed_dataset/validation \
    --output_dir finetuned_paligemma2_8bit_4_gpus \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 10 \
	--quant 8bit \
	--gpu_ids 0,1,2,3