

python3 run_mybart.py  --per_device_eval_batch_size 8 --log_root ./log --save_dataset_path dataset --exp_name attack --do_predict --predict_with_generate True --output_dir das --max_target_length 80 --model_name_or_path [model_name]
