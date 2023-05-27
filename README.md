# Improving the Robustness of Summarization Systems with Dual Augmentation (ACL 2023)

## 1. How to Install

### Requirements
- `python3`
- `conda create --name env `
- `pip3 install -r requirements.txt`

### Description of Codes
`Attack` and `Augmentation` are two directories for generating attacking datasets and training robust summarization model, respectively.
In each directory:

- `run_mybart` -> training and evaluation procedure
- `magic_bart.py` -> main models
- `dataset_maker.py` -> data preprocessing

### Workspace
`./log/seq2seqV4/` will be created for storing model checkpoints and scores.

## 2. Attacking
For attacking process, go to `Attack` directory.
Firstly, run `datasets/from_port.py` to obtain the dataset.
Then run the datamaking and decoding command to obtain attention and gradients:

```
python3 run_mybart.py --model_name_or_path facebook/bart-base\
   --do_train --do_eval --train_file [train_file] \
   --validation_file [valid_file] --test_file [test_file] --output_dir das\ 
   --exp_name [exp_name] --max_source_length 1024 --max_target_length 100 \
   --gene_dataset_path [data_name]
   
python3 run_mybart.py  --per_device_eval_batch_size 8 \
   --log_root ./log --save_dataset_path dataset --exp_name [exp_name] \
   --do_predict --predict_with_generate True --output_dir das \
   --max_target_length 120 --model_name_or_path [model_path]

```
Finally, run the attacking code: `python robust/attack.py`.

We also provide the link to attacked dataset: [https://drive.google.com/file/d/1BP5x0bhnq7eSYTc6sX5rXijrYqVTwFSz/view?usp=sharing](https://drive.google.com/file/d/1BP5x0bhnq7eSYTc6sX5rXijrYqVTwFSz/view?usp=sharing)

## 3. Augmentation
After obtaining the attacked datasets, we go to the `Augmentation` directory. We first run the datamaker code to process the data:

```
CUDA_VISIBLE_DEVICES=0 python3 run_mybart.py --model_name_or_path facebook/bart-base \
   --do_train --do_eval --train_file [train_file] \
   --validation_file [valid_file] \
   --test_file [test_file] --output_dir das \
   --exp_name [exp_name] --max_source_length 1024 \
   --max_target_length 120 --gene_dataset_path [data_name] 
```


### Train
```
python3 run_mybart.py --model_name_or_path facebook/bart-large \
     --do_train --output_dir das \
     --exp_name [exp_name] \
     --max_source_length 1024 --max_target_length 100 \
     --save_dataset_path [data_path]\
     --num_train_epochs 100 \
     --per_device_train_batch_size 8 --save_strategy epoch \
     --label_smoothing_factor 0.1 --weight_decay 0.01 \
     --max_grad_norm 0.1 --warmup_steps 500\
     --gradient_accumulation_steps 4 \
     --learning_rate 3e-05 \
     --maniED_model True --maniED_loss True
```
### Evaluate
```
python3 run_mybart.py --per_device_eval_batch_size 32 \
   --log_root ./log --save_dataset_path [data_path] \
   --exp_name [exp_name] --do_predict \
   --predict_with_generate True \
   --output_dir das \
   --val_max_target_length 120 \
   --model_name_or_path [model_path]
```

## Citation
We appreciate your citation if you find our work beneficial.

```
@article{chen2023improving,
  title={Improving the Robustness of Summarization Systems with Dual Augmentation},
  author={Chen, Xiuying and Long, Guodong and Tao, Chongyang and Li, Mingzhe and Gao, Xin and Chengqi, Zhang and Zhang, Xiangliang},
  journal={ACL},
  year={2023}
}
```
