accelerate launch --config_file src/Configs/megatron_lm_t5_config.yaml \
src/Training/run_translation_no_trainer.py \
--model_name_or_path t5-base \
--source_lang en \
--target_lang de \
--max_source_length 512 \
--max_target_length 128 \
--pad_to_max_length True \
--learning_rate 1.2e-4 \
--source_prefix "translate English to German: " \
--dataset_name stas/wmt14-en-de-pre-processed \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--output_dir "/tmp/test_megatron_lm_t5" \
--checkpointing_steps "epoch" \
--with_tracking \
--report_to "wandb" \
--n_train 10000 \
--n_val 2000 