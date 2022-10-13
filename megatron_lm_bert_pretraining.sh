accelerate launch --config_file src/Configs/megatron_lm_bert_config.yaml \
src/Training/run_mlm_no_trainer.py \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--tokenizer_name "nvidia/megatron-bert-uncased-345m" \
--pad_to_max_length \
--max_seq_length 512 \
--learning_rate 1.2e-4 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--num_train_epochs 5 \
--with_tracking \
--report_to "wandb" \
--output_dir "/tmp/test_megatron_lm_bert"
--checkpointing_steps "epoch"