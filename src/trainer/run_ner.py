# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import json
import logging
import argparse
from os.path import join as os_join
from typing import List, Union
from dataclasses import dataclass

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AutoConfig, AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizer,
    AdamW, get_linear_schedule_with_warmup
)

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.trainer.utils_ner import *
from src.trainer.eval import evaluate

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# logger = logging.getLogger(__name__)
logger = get_logger(__name__)


__all__ = ['train', 'run_train', 'TrainDummyArgs']


@dataclass
class TrainDummyArgs:
    """
    dummy args to map relevant arguments (see `scripts/train.py`) to the more detailed arguments for `run_train`
    """
    # include all arguments for completeness
    data_dir: str = None
    dataset_name: str = None
    model_type: str = None
    output_dir: str = None
    config_name: str = None
    tokenizer_name: str = None
    task: str = None
    prefix: str = None
    train_file: str = None
    validation_file: str = None
    test_file: str = None
    few_file: str = None
    unlabeled_file: str = None
    cache_dir: str = None
    cache_datasets: int = None
    max_seq_length: int = 144
    do_train: int = 1
    do_eval: int = 0
    do_predict: int = 1
    eval_no_verbose: int = 1
    eval_set_from_train: int = 0
    do_lower_case: bool = False
    gpu: str = ''
    per_gpu_train_batch_size: int = 24
    per_gpu_eval_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    max_grad_norm: float = 1.0
    num_train_epochs: float = 16.0
    max_steps: int = -1
    warmup_steps: int = 200
    logging_steps: int = 50
    eval_per_epoch: int = 1
    ignore_new_labels_in_eval: bool = False
    save_steps: int = 0
    save_per_epoch: int = 1
    best_model_split: str = None
    save_checkpoints: int = 1
    eval_all_checkpoints: bool = False
    no_cuda: bool = False
    overwrite_output_dir: bool = False
    overwrite_cache: bool = False
    seed: int = 42
    local_rank: int = -1
    train_data_source: str = 'generated'
    load_weak: bool = False
    load_unlabeled: int = 0
    remove_labels_from_weak: bool = False
    rep_train_against_weak: int = 5

    @classmethod
    def from_script(cls, args: argparse.Namespace) -> 'TrainDummyArgs':
        dnm = args.dataset_name
        dnm_ = dnm.replace('-', '_')
        gen_data_dir_nm = args.generated_dataset_dir_name
        if not os.path.exists(gen_data_dir_nm):
            gen_data_dir_nm = os_join('generated_data', dnm_, args.generated_dataset_dir_name)
        assert os.path.exists(gen_data_dir_nm), f'Generated dataset directory not found: {gen_data_dir_nm}'
        ori_data_dir_nm = os_join('original-dataset', dnm)

        return cls(
            data_dir=gen_data_dir_nm,
            dataset_name=dnm,
            model_type=args.hf_model_name,
            task=dnm,
            train_file='bio-train.jsonl',
            test_file=os_join(ori_data_dir_nm, args.test_file),
            few_file=os_join(ori_data_dir_nm, args.few_shot_demo_file),
            per_gpu_train_batch_size=args.train_batch_size,
            per_gpu_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.n_epochs,
            save_steps=999_999 if args.save_trained else 0,  # effectively save once at the end of training, and no save during training
            save_per_epoch=0,
            best_model_split='final',
            seed=args.seed,
            load_weak=True,
            rep_train_against_weak=args.demo_weight
        )


def train(
        args: argparse.Namespace = None,
        train_dataset: TensorDataset = None,
        valid_dataset: TensorDataset = None,
        test_dataset: TensorDataset = None,
        model: torch.nn.Module = None,
        tokenizer: PreTrainedTokenizer = None,
        labels: List[str] = None,
        pad_token_label_id: int = None,
        test_texts: List[List[str]] = None
):
    """
    :param args:  Command line arguments
    :param train_dataset: Training dataset
    :param valid_dataset: Validation dataset
    :param test_dataset: Test dataset
    :param model: NER model
    :param tokenizer: Tokenizer
    :param labels: List of NER tag labels in BIO format
    :param pad_token_label_id: Label id for padding
    :param test_texts: Texts as list of tokens for the test dataset, intended for logging predictions
    :return:
    """
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter(os.path.join(args.output_dir,'tfboard'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info(
    #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #     args.train_batch_size
    #     * args.gradient_accumulation_steps
    #     * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    # )
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)
    step_per_epoch = len(train_dataloader)
    d_log = {
        '#examples': len(train_dataset), '#epochs': args.num_train_epochs, 'batch size/gpu': args.per_gpu_train_batch_size,
        'total batch size': args.train_batch_size * args.gradient_accumulation_steps * (
            torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        'gradient accumulation steps': args.gradient_accumulation_steps, '#optimization steps': t_total,
        '#step/epoch': step_per_epoch, '#warmup steps': args.warmup_steps,
        'best-model-split': args.best_model_split, 'save-checkpoints': args.save_checkpoints
    }
    if args.eval_per_epoch > 0 and args.logging_steps != step_per_epoch:
        logger.warning(f'Logging steps changed: {pl.i(args.logging_steps)} -> {pl.i(step_per_epoch)}')
        args.logging_steps = step_per_epoch
    if args.save_per_epoch > 0 and args.save_steps > 0 and args.save_steps != step_per_epoch:
        logger.warning(f'Save steps changed: {pl.i(args.save_steps)} -> {pl.i(step_per_epoch)}')
        args.save_steps = step_per_epoch
    if args.save_steps > 0 and args.best_model_split == 'test':
        logger.warning(f'Best model will be selected based on {pl.i("test")} set')

    ignore_labels = None
    if args.ignore_new_labels_in_eval:
        if args.dataset_name == 'conll2003':
            ori_labels = ['person', 'location', 'organization', 'miscellaneous']
        else:
            raise NotImplementedError
        ori_tags = ner_labels2tags(ori_labels)
        assert ori_tags[0] == 'O'
        ignore_labels = sorted(set(labels) - set(ori_tags))
        d_log['ignore labels'] = ignore_labels
    dnm = args.dataset_name.replace('_', '-')
    entity_types = sconfig(f'datasets.{dnm}.readable-entity-types')

    logger.info(f'***** Running training *****\n{pl.fmt(d_log)}')

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):  # TODO: not supported for now cos this variable not in namespace?
        try:
            # set global_step to global_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except:
            logger.warning(f"Unable to recover training step from {args.model_name_or_path}")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0], unit='ep')
    set_seed(args)  # Added here for reproducibility
    test_results, best_dev, best_test, test_by_best_val = None, None, None, None
    # best_dev_f1 = dict(epoch=0, step=0, f1=-math.inf)
    # best_test_f1 = dict(epoch=0, step=0, f1=-math.inf)

    mp = MlPrettier(ref=dict(step=step_per_epoch, epoch=int(args.num_train_epochs), global_step=int(t_total)))
    tb_writer = SummaryWriter(os_join(args.output_dir, 'tensorboard'))  # TODO: support multi-gpu training
    ls_args = dict(logger=logger, file_logger=True, prettier=mp, tb_writer=tb_writer)
    ls = LogStep(**ls_args, prettier_console=True, console_with_split=True, global_step_with_epoch=False)

    for epoch in train_iterator:
        desc = f"Train {pl.i(epoch+1)}/{pl.i(int(args.num_train_epochs))}"
        epoch_iterator = tqdm(train_dataloader, desc=desc, disable=args.local_rank not in [-1, 0], unit='ba')
        ls.pbar = epoch_iterator
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "biobert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            loss_item = loss.item()
            tr_loss += loss.item()

            lr = scheduler.get_last_lr()[0]
            d_log = dict(epoch=epoch, step=global_step, lr=lr, loss=loss_item)
            ls(d_log, training=True, to_console=False)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.eval_no_verbose:
                        verb = global_step == args.logging_steps  # log only the 1st one
                    else:
                        verb = True

                    # Log metrics
                    t_prefix = (f'[Step {pl.i(global_step)}/{pl.i(int(t_total))} | '
                                f'Epoch {pl.i(epoch + 1)}/{pl.i(int(args.num_train_epochs))}]')
                    dev_is_updated, test_is_updated = None, None
                    ls_args = dict(training=False, add_pbar_postfix=False)
                    eval_args = dict(
                        args=args, model=model, labels=labels, pad_token_label_id=pad_token_label_id,
                        ignore_labels=ignore_labels, entity_types=entity_types, verbose=verb, step=global_step, epoch=epoch, logger=logger)
                    if args.do_eval:
                        dev_results, best_dev, dev_is_updated = evaluate(
                            **eval_args, best=best_dev, eval_dataset=valid_dataset, prefix=f'{pl.i("dev")} {t_prefix}', split='dev')
                        d_log = dict(global_step=global_step)
                        d_log.update(dev_results.to_performance_dict())
                        d_log.update(best_dev.to_performance_dict(prefix='best', keys=['loss', 'f1']))
                        ls(d_log, prefix=f'Dev {t_prefix} results\n', split='dev', **ls_args)

                    if args.do_predict:
                        test_results, best_test, test_is_updated = evaluate(
                            **eval_args, best=best_test, eval_dataset=test_dataset,
                            prefix=f'{pl.i("test")} {t_prefix}', texts=test_texts, split='test')
                        d_log = dict(global_step=global_step)
                        d_log.update(test_results.to_performance_dict())  # override for the same keys
                        d_log.update(best_test.to_performance_dict(prefix='best', keys=['loss', 'f1']))
                        ls(d_log, prefix=f'Test {t_prefix} results\n', split='test', **ls_args)

                        if args.do_eval and dev_is_updated:
                            test_by_best_val = test_results  # save the test score from the best dev model

                    if args.do_eval or args.do_predict:
                        output_dirs = []
                        best_split = args.best_model_split
                        if not best_split or best_split == 'final':  # the final model is the last checkpoint, always save
                            save_final = True
                        else:  # override if better metric
                            save_final = dev_is_updated if best_split == 'dev' else test_is_updated
                        if args.save_steps > 0:
                            if args.save_checkpoints and global_step % args.save_steps == 0:
                                output_dirs.append(os.path.join(args.output_dir, "checkpoint-{}".format(global_step)))
                            if args.local_rank in [-1, 0] and save_final:  # the final saved model will be the best model
                                output_dirs.append(os.path.join(args.output_dir, "trained"))

                        for output_dir in output_dirs:
                            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                            # They can then be reloaded using `from_pretrained()`
                            os.makedirs(output_dir, exist_ok=True)
                            # Take care of distributed/parallel training
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info(f"Model and tokenizer checkpoints saved to {pl.i(output_dir)}")

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    d_log = dict()
    if args.do_eval:
        d_log['best-dev-f1'] = mp(best_dev.to_f1_log())
    d_log['final-test-f1'] = mp(test_results.to_f1_log())
    if args.do_predict:
        d_log['best-test-f1'] = mp(best_test.to_f1_log())
        if args.do_eval:
            assert test_by_best_val is not None
            d_log['test-f1-by-best-val'] = mp(test_by_best_val.to_f1_log())
    logger.info(f'Micro f1 scores: {pl.i(d_log, indent=1)}')
    return global_step, tr_loss / global_step, best_dev, best_test


def run_train(args: Union[argparse.Namespace, TrainDummyArgs] = None):
    args.output_dir = args.output_dir or get_output_dir(args=args)
    args.cache_dir = args.cache_dir or os_join(pu.model_path, args.task, 'cache')

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # Setup distant debugging if needed
    # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl")
    #     args.n_gpu = 1
    args.device = device

    if 'deberta-base' in args.model_type:
        args.model_name_or_path = 'microsoft/deberta-v3-base'
        args.tokenizer = 'microsoft/deberta-v3-base'
    elif 'deberta-large' in args.model_type:
        args.model_name_or_path = 'microsoft/deberta-v3-large'
        args.tokenizer = 'microsoft/deberta-v3-large'
    elif args.model_type == 'pubmedbert':
        args.model_name_or_path = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        args.tokenizer = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    else:
        args.model_name_or_path = args.model_type
        args.tokenizer = args.model_type
    # this will not work here since torch already imported: https://github.com/pytorch/pytorch/issues/9158
    gpus = os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    d_log = dict(model_name_or_path=args.model_name_or_path, tokenizer=args.tokenizer, gpus=gpus.split(','))
    logger.info(f'Using {pl.i(d_log)}')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    # logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    # logging_fh.setLevel(logging.DEBUG)
    # logger.addHandler(logging_fh)
    add_file_handler(logger=logger, file_path=os.path.join(args.output_dir, 'run_ner.log'))
    d_log = {'process rank': args.local_rank, 'device': device, '#gpu': args.n_gpu, 'distributed training': args.local_rank != -1}
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
    #     args.local_rank,
    #     device,
    #     args.n_gpu,
    #     bool(args.local_rank != -1),
    #     # args.fp16,
    # )
    logger.warning(pl.i(d_log))

    # Set seed
    set_seed(args)
    labels = get_ner_tags(path=args.data_dir, dataset_name=args.dataset_name)
    num_labels = len(labels)
    d_log = {'dataset-name': args.task, '#labels': num_labels, 'labels': labels}

    path = os_join(args.data_dir, 'tag_to_id.json')
    if os.path.exists(path):
        msg = f'Loaded labels from {pl.i(path)}'
    else:
        msg = f'Loaded default labels'
    logger.info(f'{msg} w/ {pl.i(d_log)}')

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    fast_tokenizer = True
    if 'deberta' in args.model_type:  # TODO: doesn't seem to influence performance?
        """
        For warning:
        /nethome/yheng6/miniconda3/envs/llm-ie-gen/lib/python3.8/site-packages/transformers/convert_slow_tokenizer.py:473: UserWarning: 
        The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option 
        which is not implemented in the fast tokenizers. 
        In practice this means that the fast version of the tokenizer can produce unknown tokens 
        whereas the sentencepiece version would have converted 
        these unknown tokens into a sequence of byte tokens matching the original piece of text.
            warnings.warn(
        Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
        """
        fast_tokenizer = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=fast_tokenizer,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    args_log = copy.deepcopy(vars(args))
    args_log['device'] = args_log['device'].__repr__()
    logger.info(f"Training/evaluation parameters {pl.fmt(args_log)}")

    # Training

    tag2id = get_tag_to_id(path=args.data_dir, dataset_name=args.dataset_name, verbose=True)
    load_args = dict(args=args, tokenizer=tokenizer, labels=labels, pad_token_label_id=pad_token_label_id, tag2id=tag2id)
    if args.do_train:
        valid_dataset = None
        if args.do_eval:
            if args.validation_file:
                assert args.eval_set_from_train == 0  # sanity check
                valid_dataset = load_and_cache_examples(**load_args, mode="valid").tensor_dataset
            else:
                assert args.train_data_source == 'generated' and args.eval_set_from_train > 0
        out = load_and_cache_examples(**load_args, mode="test", return_texts=True)
        test_dataset, txts_ts = out.tensor_dataset, out.texts
        # for training
        if args.train_data_source == 'generated':  # train on generated synthetic data
            # this dataset also don't go into
            few_dataset = load_and_cache_examples(**load_args, mode="few").tensor_dataset
            # import ipdb; ipdb.set_trace()
            d_szs = dict(few=len(few_dataset))
            if args.load_weak:
                weak_dataset = load_and_cache_examples(**load_args, mode="train", remove_labels=args.remove_labels_from_weak).tensor_dataset
                d_szs['llm-generated'] = len(weak_dataset)

                if args.do_eval and args.eval_set_from_train > 0:
                    # split a val set from the generated training data
                    ratio = args.eval_set_from_train if args.eval_set_from_train < 1 else 0.1
                    logger.info(f'Splitting {pl.i(ratio)} of the generated training data as validation set')
                    assert valid_dataset is None
                    gen = torch.Generator().manual_seed(args.seed)
                    weak_dataset, valid_dataset = random_split(dataset=weak_dataset, lengths=[1 - ratio, ratio], generator=gen)

                # merge  generated training data w/ few example data from original training set, assign them higher weights by duplicating
                train_dataset = torch.utils.data.ConcatDataset([few_dataset]*args.rep_train_against_weak + [weak_dataset])
            else:
                train_dataset = []  # TODO: should be just the `few_dataset` instead of empty?
            if args.load_unlabeled:
                unlabeled_dataset = load_and_cache_examples(**load_args, mode="unlabeled").tensor_dataset
                d_szs['unlabeled-corpus'] = len(unlabeled_dataset)
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, unlabeled_dataset])
            d_szs = dict(train=len(train_dataset), **d_szs)
        else:
            train_dataset = load_and_cache_examples(**load_args, mode="train").tensor_dataset
            d_szs = dict(train=len(train_dataset))
        if valid_dataset:
            d_szs['val'] = len(valid_dataset)
        d_szs['test'] = len(test_dataset)
        logger.info(f'Dataset sizes: {pl.i(d_szs)}')

        t = Timer()
        global_step, tr_loss, best_dev, best_test = train(
            args=args, train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, model=model,
            tokenizer=tokenizer, labels=labels, pad_token_label_id=pad_token_label_id,
            test_texts=txts_ts
        )
        d_log = {'global step': global_step, 'average loss': round_f(tr_loss)}
        logger.info(f'Training finished w/ {pl.i(d_log)} in {pl.i(t.end())}')

        # save classification reports
        for split, res in zip(['dev', 'test'], [best_dev, best_test]):
            if res:
                res.write_reports()
        # save predictions on the test set
        if best_test:
            best_test.write_predictions()
            logger.info(f'Test set best entity-wise f1: {pl.i(best_test.entity_wise_f1())}')

        with open(os_join(pu.model_path, args.task, 'result.jsonl'), 'a') as f:
            result = dict()
            if best_dev:
                result['dev'] = {k: round_metric(m) for k, m in best_dev.to_performance_dict().items()}
            if best_test:
                result['test'] = {k: round_metric(m) for k, m in best_test.to_performance_dict().items()}

            result.update(prefix=args.prefix, model=args.model_type, lr=args.learning_rate)
            f.write(json.dumps(result) + '\n')
        logger.info(f'Eval results saved to {pl.i(stem(args.output_dir, top_n=2))}')
    return


if __name__ == "__main__":
    def main():
        parser = argparse.ArgumentParser()

        # Required parameters
        parser.add_argument(
            "--data_dir", default=None, type=str, required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
        parser.add_argument("--dataset_name", default=None, type=str, required=True, help="Dataset name")
        parser.add_argument("--model_type", default=None, type=str, required=True, )
        # parser.add_argument(
        #     "--model_name_or_path",
        #     default=None,
        #     type=str,
        #     required=True,
        #     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
        # )
        parser.add_argument(
            "--output_dir", default=None, type=str, required=False,
            help="The output directory where the model predictions and checkpoints will be written.", )

        parser.add_argument(
            "--config_name", default="", type=str,
            help="HuggingFace pretrained config name or path if not the same as model_name")
        parser.add_argument(
            "--tokenizer_name", default="", type=str,
            help="HuggingFace pretrained tokenizer name or path if not the same as model_name", )

        parser.add_argument("--task", required=True, type=str, help="the task/dataset name", )

        parser.add_argument(
            "--prefix", default="", type=str, help="the task prefix",
        )

        parser.add_argument("--train_file", default="", type=str, help="path to jsonl file containing training data", )

        parser.add_argument("--validation_file", default="", type=str, help="path to jsonl file containing validation data", )

        parser.add_argument("--test_file", default="", type=str, help="path to jsonl file containing test data", )

        parser.add_argument("--few_file", default="", type=str, help="path to jsonl file containing few data", )

        parser.add_argument(
            "--unlabeled_file", default="", type=str,
            help="path to jsonl file containing data from unlabeled in-domain corpus", )

        parser.add_argument(
            "--cache_dir", default="", type=str,
            help="Where do you want to store the pre-trained models downloaded from s3", )
        parser.add_argument('--cache_datasets', default=1, type=int, help='Cache datasets or not')
        parser.add_argument(
            "--max_seq_length", default=144, type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter than this will be padded.", )
        parser.add_argument("--do_train", default=1, type=int, help="Whether to run training.")
        parser.add_argument("--do_eval", default=1, type=int, help="Whether to run eval on the dev set.")
        parser.add_argument("--do_predict", default=1, type=int, help="Whether to run predictions on the test set.")
        parser.add_argument("--eval_no_verbose", default=1, type=int, help="Disable eval verbose logging")
        parser.add_argument(
            "--eval_set_from_train", default=0., type=float,
            help="If given, use a subset of training set as dev set. If < 1, use the given fraction of training set as dev set.")
        parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
        parser.add_argument("--gpu", help="Set this flag if you are using an uncased model.")

        parser.add_argument("--per_gpu_train_batch_size", default=24, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument(
            "--gradient_accumulation_steps", type=int, default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.", )
        parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
        parser.add_argument("--adam_beta2", default=0.98, type=float, help="BETA2 for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument("--num_train_epochs", default=16.0, type=float, help="Total number of training epochs to perform.")
        parser.add_argument(
            "--max_steps", default=-1, type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
        parser.add_argument("--warmup_steps", default=200, type=int, help="Linear warmup over warmup_steps.")

        parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps where eval is run.")
        parser.add_argument("--eval_per_epoch", type=int, default=1,
                            help="If given, override `logging_steps` to number of steps per epoch.")
        parser.add_argument(
            '--ignore_new_labels_in_eval', action="store_true",
            help='If given, additional labels added from synthetic dataset will be ignored in evaluation')
        parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
        parser.add_argument("--save_per_epoch", type=int, default=1,
                            help="If given and save checkpoints, override `save_steps` to number of steps per epoch.")
        parser.add_argument('--best_model_split', type=str, default=None,
                            help='Use dev or test to select best model. If final is given, save the final trained model',
                            choices=['final', 'dev', 'test'])
        parser.add_argument(
            '--save_checkpoints', type=int, default=1,
            help='Save model checkpoints or not. If set to 0, override `save_steps`, only the best model will be saved.')
        parser.add_argument(
            "--eval_all_checkpoints", action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
        parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
        parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
        parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

        # parser.add_argument(
        #     "--fp16",
        #     action="store_true",
        #     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        # )
        # parser.add_argument(
        #     "--fp16_opt_level",
        #     type=str,
        #     default="O1",
        #     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        #     "See details at https://nvidia.github.io/apex/amp.html",
        # )
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        parser.add_argument(
            "--train_data_source", type=str, default='generated',
            help="Train with synthetic generated data or original data", choices=['generated', 'original'])

        # Use data from weak.json
        parser.add_argument('--load_weak', action="store_true", help='Load data from weak.json.')  # TOOD: what is this?
        parser.add_argument('--load_unlabeled', type=int, default=0, help='IF >0, load data from unlabeled in-domain corpus.')
        parser.add_argument('--remove_labels_from_weak', action="store_true",
                            help='Use data from weak.json, and remove their labels for semi-supervised learning')
        parser.add_argument('--rep_train_against_weak', type=int, default=5, help='Upsampling training data again weak data. Default: 5')

        args = parser.parse_args()
    main()
