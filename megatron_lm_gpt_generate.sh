accelerate launch --config_file src/Configs/megatron_lm_gpt_generate_config.yaml \
src/inference/megatron_gpt2_generation.py \
--config_name "gpt2" \
--tokenizer_name "gpt2" \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--config_overrides "vocab_size=50257,n_embd=1024,n_head=16,n_layer=24,n_positions=1024" \
--block_size 1024 \
--learning_rate 5e-5 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--num_train_epochs 2 \
--output_dir "awesome_model" \
--resume_from_checkpoint "/home/sourab/temp/megatron_lm_checkpoint" \
--with_tracking \
--report_to "wandb" \
--n_train 0 \
--n_val 0
