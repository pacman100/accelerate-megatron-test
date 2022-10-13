accelerate launch --config_file src/Configs/fsdp_gpt_config.yaml  \
src/Training/run_clm_no_trainer.py \
--config_name "gpt2" \
--tokenizer_name "gpt2" \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--config_overrides "vocab_size=50257,n_ctx=2048,n_embd=4096,n_head=16,n_layer=32,n_positions=2048" \
--block_size 1024 \
--learning_rate 1.2e-4 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--num_train_epochs 1 \
--with_tracking \
--report_to "wandb" \
--output_dir "/tmp/test_fsdp_gpt" \
--n_train 0 \
--n_val 0 \
--checkpointing_steps "epoch"
