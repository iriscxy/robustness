

python3 run_mybart.py --model_name_or_path facebook/bart-base --do_train --do_eval --train_file train.json --validation_file validation.json --test_file test.json --output_dir das --exp_name [dataset_name] --max_source_length 1024 --max_target_length 100 --gene_dataset_path [dataset_name]





