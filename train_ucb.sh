# accelerate launch --mixed_precision="fp16" --gpu_ids="0" train_text_to_image_lora_cub.py \
#   --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
#   --train_data_dir=dataset \
#   --caption_column="caption" --image_column="image" \
#   --resolution=512 --random_flip \
#   --train_batch_size=4 --gradient_accumulation_steps=4 \
#   --allow_tf32 --enable_xformers_memory_efficient_attention \
#   --num_train_epochs=50 --checkpointing_steps=500 \
#   --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --seed=42 \
#   --output_dir="output/stabilityai/stable-diffusion-2-1/ft_lora_coarse" \
#   --validation_prompt="Mourning Warbler" --report_to="wandb"
accelerate launch --mixed_precision="fp16" --gpu_ids="0" train_text_to_image_lora_cub.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --train_data_dir=dataset \
  --caption_column="caption" --image_column="image" \
  --resolution=512 --random_flip \
  --train_batch_size=4 --gradient_accumulation_steps=4 \
  --allow_tf32 --enable_xformers_memory_efficient_attention \
  --num_train_epochs=10 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 -t fine \
  --output_dir="output/stabilityai/stable-diffusion-2-1/ft_lora_fine" \
  --validation_prompt="Mourning Warbler" --report_to="wandb"