import logging
import pdb
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from args import DataTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
import random
from datasets import load_dataset, DownloadConfig
import numpy as np

logger = logging.getLogger(__name__)
prob = 0.6


def random_middle(w, probability=prob):
    """
      Randomly permute the middle of a word (all but first and last char)
    """
    if random.random() > probability:
        return w
    if len(w) > 3:
        middle = list(w[1:len(w) - 1])
        random.shuffle(middle)
        middle = ''.join(middle)
        return w[0] + middle + w[len(w) - 1]
    else:
        return w


def swap(w, probability=prob):
    if random.random() > probability:
        return w
    if len(w) > 3:
        w = list(w)
        i = random.randint(1, len(w) - 3)
        w[i], w[i + 1] = w[i + 1], w[i]
        return ''.join(w)
    else:
        return w



class DatasetMaker:
    def __init__(self, dataset_saved_path: str, data_args: DataTrainingArguments,
                 training_args: Seq2SeqTrainingArguments, tokenizer: PreTrainedTokenizerBase):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.dataset_saved_path = dataset_saved_path

    def make_dataset(self):
        logger.info('******* Making Dataset **********')
        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if self.data_args.test_file is not None:
            data_files["test"] = self.data_args.test_file
            extension = self.data_args.test_file.split(".")[-1]
        if extension == 'txt': extension = 'text'
        datasets = load_dataset(extension, data_files=data_files, download_config=DownloadConfig(use_etag=False))
        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = "max_length" if self.data_args.pad_to_max_length else False

        if self.training_args.label_smoothing_factor > 0:
            logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for model. This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples: Dict):
            """
            如果是json，examples就是json对应的dict。如果是纯文本，examples["text"]就是全部文本,每个item就是文本文件中的一行
            """
            if isinstance(examples["src"][0], str):
                inputs = [ex.replace(' ', '') if self.data_args.chinese_data else ex for ex in examples["src"]]
            elif isinstance(examples["src"][0], list):
                inputs = [' '.join(ex).replace(' ', '') if self.data_args.chinese_data else ' '.join(ex) for ex in
                          examples["src"]]
            else:
                raise ValueError(f'only support str/list in content, now {type(examples["src"][0])}')


            if isinstance(examples["tgt"][0], str):
                targets = [ex.replace(' ',
                                      '') + self.tokenizer.eos_token if self.data_args.chinese_data else ex + self.tokenizer.eos_token
                           for ex in examples["tgt"]]
            elif isinstance(examples["tgt"][0], list):
                targets = [' '.join(ex).replace(' ',
                                                '') + self.tokenizer.eos_token if self.data_args.chinese_data else ' '.join
                                                                                                                   (ex) + self.tokenizer.eos_token
                           for ex in examples["tgt"]]
            else:
                raise ValueError(f'only support str/list in summary, now {type(examples["tgt"][0])}')

            model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                          truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        datasets = datasets.map(
            preprocess_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        logger.info('saving dataset')
        dataset_saved_path = self.dataset_saved_path
        datasets.save_to_disk(dataset_saved_path)
        logger.info(f'******* Dataset Finish {dataset_saved_path} **********')
        return datasets
