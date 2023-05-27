#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for sequence to sequence.
"""
import logging
import pdb
import sys
import traceback
# import comet
from dataset_maker import DatasetMaker
import glob
import zipfile

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
import os

# from magic_bart import MyBart, MyCometCallback, AutoDecodeCallback, MyDataCollatorForSeq2Seq, MySeq2SeqTrainer,MyBartConfig
from magic_bart import MyBart, MyDataCollatorForSeq2Seq, MySeq2SeqTrainer
from transformers import BartForCausalLM

import nltk  # Here to have a nice missing dependency error message early on
from datasets import DatasetDict

import transformers
from filelock import FileLock
from transformers import (
    HfArgumentParser,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.models.bart.tokenization_bart import BartTokenizer
from compute_metric import MetricCompute
from args import ModelArguments, DataTrainingArguments, my_Seq2SeqTrainingArguments

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, my_Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # type: ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments
    training_args.logging_steps = 10
    data_args.log_root = os.path.join(data_args.log_root, data_args.proj_name, data_args.exp_name)

    training_args.output_dir = os.path.join(data_args.log_root, 'model')
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    if training_args.do_train:
        python_list = glob.glob('./*.py')
        zip_file = zipfile.ZipFile(data_args.log_root + '/code.zip', 'w')
        for d in python_list:
            zip_file.write(d)
        for d in glob.glob('dataset/*.py'):
            zip_file.write(d)
        for d in glob.glob('cmd/*.py'):
            zip_file.write(d)
        for d in glob.glob('metrics/*.py'):
            zip_file.write(d)
        zip_file.close()
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    # logger.info("Training/evaluation parameters %s", training_args)
    # logger.info("Dataset parameters %s", data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not training_args.do_train and (
            training_args.do_eval or training_args.do_predict) and model_args.model_name_or_path is None:
        # 纯测试且没指定ckpt 就用最新的ckpt
        model_args.model_name_or_path = last_checkpoint if last_checkpoint is not None else get_last_checkpoint(
            training_args.output_dir)
    if training_args.do_train and last_checkpoint is not None:
        logger.warning(f'using previous checkpoint {last_checkpoint}')
        model_args.model_name_or_path = last_checkpoint

    logger.info(f'******* Loading model form pretrained {model_args.model_name_or_path} **********')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')  # 如果用bart-base就用这行
    logger.info('load BartTokenizer')

    model = MyBart.from_pretrained(model_args.model_name_or_path)

    model.config.margin_model = model_args.margin_model
    model.config.kl_model = model_args.kl_model
    model.config.output_attentions=True
    model.config.return_dict_in_generate=True

    logger.info('load model')

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if data_args.save_dataset_path is None and data_args.gene_dataset_path:
        maker = DatasetMaker(data_args.gene_dataset_path, data_args, training_args, tokenizer)
        datasets = maker.make_dataset()
    else:
        logger.info(f'******* Loading Dataset from {data_args.save_dataset_path} **********')
        datasets = DatasetDict.load_from_disk(data_args.save_dataset_path)

    train_dataset = datasets["test"]
    eval_dataset = datasets["validation"] if training_args.do_eval is not None and "validation" in datasets else None
    test_dataset = datasets["test"]
    if training_args.do_predict is None and "test" not in datasets:
        logging.warning(f'using validation dataset as test!')

    if data_args.max_val_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_val_samples))

    max_target_length = data_args.val_max_target_length
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = MyDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    comp_metric = MetricCompute(data_args, tokenizer, test_dataset, eval_dataset)

    # comet_callback = MyCometCallback(data_args.proj_name, data_args.exp_name)

    model.config.num_beams = data_args.num_beams
    model.config.max_length = data_args.max_target_length
    # for param in model.parameters():
    #     param.requires_grad = False

    # for arg_class in [model_args, data_args, training_args, model.config]:
    #     for k, v in arg_class.to_dict().items():
    #         comet_callback.exp.experiment.log_parameter(k, v)
    # python_list = glob.glob('./*.py')
    # for file in python_list:
    #     comet_callback.exp.experiment.log_code(file_name=file, folder='./', code=None, code_name=None)

    # Initialize our Trainer
    if training_args.predict_with_generate:
        training_args.report_to = ['comet_ml']
    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=comp_metric.compute_metrics if training_args.predict_with_generate else None,
        # callbacks=[comet_callback]  # auto_decode_callback
    )
    comp_metric.trainer = trainer
    # comet_callback.set_trainer(trainer)

    # Training
    if training_args.do_train:
        try:
            if last_checkpoint is not None:  # 如果是继续之前的训练需要加载步数和optimizer
                train_result = trainer.train(
                    resume_from_checkpoint=model_args.model_name_or_path)  # resume_from_checkpoint=checkpoint
            else:
                train_result = trainer.train()
            logger.info("***** Train results *****")
            # for key, value in sorted(train_result.metrics.items()):
            #     logger.info(f"  {key} = {value}")
        except KeyboardInterrupt:
            logger.info('stop training')
        finally:
            traceback.print_exc()
            if trainer.is_world_process_zero():
                logger.info('exit, saving model')
                trainer.save_model(output_dir=os.path.join(training_args.output_dir,
                                                           f'checkpoint-{trainer.state.global_step}'))  # Saves the tokenizer too for easy upload
                trainer.state.save_to_json(
                    os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}',
                                 'trainer_state.json'))
            exit(0)

    # predict
    if training_args.do_predict:
        logger.info(f"*** Test ***")
        trainer.state.global_step = model_args.model_name_or_path.split('-')[-1]
        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        print(test_results.metrics)
        # for k, v in test_results.metrics.items():
        #     comet_callback.exp.experiment.log_metric(name=k, value=v, step=int(trainer.state.global_step))

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_results.label_ids[test_results.label_ids < 0] = tokenizer.pad_token_id
                test_label = tokenizer.batch_decode(
                    test_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                test_labels = [label.strip() for label in test_label]
                # for pred, lab in zip(test_preds[:10], test_labels[:10]):
                #     logger.info(f'{pred}\t{lab}')

                dec_dir = os.path.join(data_args.log_root, f'decode-{trainer.state.global_step}')
                if not os.path.exists(dec_dir):
                    os.makedirs(dec_dir)
                fo_ref = open(os.path.join(dec_dir, 'reference.txt'), 'w', encoding='utf8')
                fo_dec = open(os.path.join(dec_dir, 'decoded.txt'), 'w', encoding='utf8')
                for pred, lab in zip(test_preds, test_labels):
                    fo_ref.write(f'{lab}\n')
                    fo_dec.write(f'{pred}\n')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    with open('mybart.pid', 'w', encoding='utf8') as w:
        w.write(str(os.getpid()))
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()
