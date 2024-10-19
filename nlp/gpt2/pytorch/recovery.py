#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import datetime
import json
import logging
import math
import os
import random
import time
from itertools import chain
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import datasets
import torch
# from accelerate import Accelerator, DistributedType
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import horovod.torch as hvd
import sys
sys.path.append('../..')
sys.path.append('..')
import compression
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
torch.multiprocessing.set_start_method('spawn', force=True)
import threading

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.39.0.dev0")

# logger = get_logger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
gradient_accumulation_group = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    
    # Max Epochs
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    
    parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    parser.add_argument('--compressor', type=str, default = 'topkef', help='Specify the compressors if density < 1.0')
    parser.add_argument('--memory', type=str, default = 'residual', help='Error-feedback')
    parser.add_argument('--compressor_ratio', type=float, default=1, help='Density for sparsification')
    parser.add_argument("--diff", default=True, type=bool,help='Whether to use differentail ckpt')
    parser.add_argument("--checkpoint_path", default="/data/recovery/", type=str,
                        help='the path which saves checkpoint')
    parser.add_argument('--model', type=str, default='GPT2',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--dataset', default='wikitext-2', type=str, help='dataset type')
    parser.add_argument("--fullfreq", default=10, type=int, help='Full Checkpoint Frequency')
    parser.add_argument("--resume_path", default="/data/recovery/GPT2_wikitext-2_0.01_20241007_083521/",
                        type=str, help='the path of checkpoints when resuming training')
    parser.add_argument("--gradient_accumulation", default=False, type=bool)
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def main():
    args = parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # allreduce_batch_size = args.batch_size * args.batches_per_allreduce
    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
    # cudnn.benchmark = True
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    if hvd.rank() == 0:
        args.checkpoint_path = create_checkpoint_dir(args.checkpoint_path, args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    # accelerator_log_kwargs = {}

    # if args.with_tracking:
    #     accelerator_log_kwargs["log_with"] = args.report_to
    #     accelerator_log_kwargs["project_dir"] = args.output_dir

    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)

    # Handle the repository creation
    # if accelerator.is_main_process:
    #     if args.push_to_hub:
    #         # Retrieve of infer repo_name
    #         repo_name = args.hub_model_id
    #         if repo_name is None:
    #             repo_name = Path(args.output_dir).absolute().name
    #         # Create repo and retrieve repo_id
    #         api = HfApi()
    #         repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

    #         with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
    #             if "step_*" not in gitignore:
    #                 gitignore.write("step_*\n")
    #             if "epoch_*" not in gitignore:
    #                 gitignore.write("epoch_*\n")
    #     elif args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)
    # accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    # with accelerator.main_process_first():
    #     tokenized_datasets = raw_datasets.map(
    #         tokenize_function,
    #         batched=True,
    #         num_proc=args.preprocessing_num_workers,
    #         remove_columns=column_names,
    #         load_from_cache_file=not args.overwrite_cache,
    #         desc="Running tokenizer on dataset",
    #     )
    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    # with accelerator.main_process_first():
    #     lm_datasets = tokenized_datasets.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=args.preprocessing_num_workers,
    #         load_from_cache_file=not args.overwrite_cache,
    #         desc=f"Grouping texts in chunks of {block_size}",
    #     )
    lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())    
    train_dataloader = DataLoader(
        train_dataset,  collate_fn = default_data_collator, batch_size=args.per_device_train_batch_size,
        sampler=train_sampler, **kwargs)
    
    
    # DataLoaders creation:
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    # )
    # eval_dataloader = DataLoader(
    #     eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    # )


    val_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size,
        sampler=val_sampler, **kwargs)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        # num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_warmup_steps=args.num_warmup_steps * hvd.size(),
        num_training_steps=args.max_train_steps
        # if overrode_max_train_steps
        # else args.max_train_steps * accelerator.num_processes,
    )


    ### model to cuda
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)

    # Horovod: broadcast parameters & optimizer state.
    # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )
    
    # Compression
    comm_params = {
    'comm_mode':'allgather',
    'compressor': args.compressor ,
    'compress_ratio' : args.compressor_ratio ,
    'memory':'residual',
    'send_size_aresame':True,
    'model_named_parameters': model.named_parameters(),
    }
    
    # optimizer = compression.DistributedOptimizer(optimizer, comm_params=comm_params, named_parameters=model.named_parameters())
    
    # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )

    
    

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        # accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    # total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_batch_size = args.per_device_train_batch_size * hvd.size() * args.gradient_accumulation_steps
   
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0

    if args.resume and hvd.rank() == 0:
        resume_begin = time.time()
        train_loader_len = int(len(train_dataloader))
        model, optimizer = load_base_checkpoint(model, optimizer, args)
        from compression.compressor.topk import TopKCompressor
        topk = TopKCompressor(comm_params['compress_ratio'], 0)
        if '-' in args.resume:
            if args.gradient_accumulation:
                model, optimizer = new_load_differential_checkpoint(model, optimizer, topk, args)
            else:
                model, optimizer = load_differential_checkpoint(model, optimizer, train_loader_len, topk, args)
        resume_end = time.time()
        print("Load Checkpoint from recent checkpoint takes {}s".format(resume_end - resume_begin))

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    optimizer = compression.DistributedOptimizer(optimizer, comm_params=comm_params,
                                                 named_parameters=model.named_parameters())
    # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
    #         checkpoint_path = args.resume_from_checkpoint
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
    #         dirs.sort(key=os.path.getctime)
    #         path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
    #         checkpoint_path = path
    #         path = os.path.basename(checkpoint_path)
    #
    #     # accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
    #     # accelerator.load_state(checkpoint_path)
    #
    #     # Extract `epoch_{i}` or `step_{i}`
    #     training_difference = os.path.splitext(path)[0]
    #
    #     if "epoch" in training_difference:
    #         starting_epoch = int(training_difference.replace("epoch_", "")) + 1
    #         resume_step = None
    #         completed_steps = starting_epoch * num_update_steps_per_epoch
    #     else:
    #         # need to multiply `gradient_accumulation_steps` to reflect real steps
    #         resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
    #         starting_epoch = resume_step // len(train_dataloader)
    #         completed_steps = resume_step // args.gradient_accumulation_steps
    #         resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    # progress_bar.update(completed_steps)
    verbose = 1 if hvd.rank() == 0 else 0
    
    if (hvd.rank() == 0):
        print("Start training!!!")
        print("starting_epoch: ", starting_epoch)
        print("args.num_train_epochs: ", args.num_train_epochs)
        print("args.max_train_steps: ", args.max_train_steps)
        print("len(train_dataloader): ", len(train_dataloader))
    train_begin = time.time()
    

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        with tqdm(total=len(train_dataloader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
            if args.with_tracking:
                total_loss = 0
            
            # if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            #     # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            #     active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            # else:
            #     active_dataloader = train_dataloader
            active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                # with accelerator.accumulate(model):

                ### to cuda
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                
                # accelerator.backward(loss)
                loss.backward()
               
                optimizer.step()

                
                if hvd.rank() == 0 and args.diff:
                    if step % args.fullfreq == 0:
                        filename = args.checkpoint_path + '/{}_{}_{}_{}_{}-{}_full.pth.tar'.format(args.model,
                                                                                                   args.dataset,
                                                                                                   args.compressor,
                                                                                                   args.compressor_ratio,
                                                                                                   epoch,
                                                                                                   step)
                        begin = time.time()
                        torch.save({
                            'epoch': epoch + 1,
                            'net': args.model,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, filename)
                        print("saved full checkpoint {} takes {}".format(filename, time.time() - begin))
                if hvd.rank() == 0:
                    optimizer.save_differential_checkpoint(step,
                        args.checkpoint_path + '/{}_{}_{}_{}_{}-{}_diff.pth.tar'.format(args.model, args.dataset,
                                                                                    args.compressor,
                                                                                    args.compressor_ratio, epoch,
                                                                                    step))

                t.update(1)
                completed_steps += 1

                # Checks if the accelerator has performed an optimization step behind the scenes
                # if accelerator.sync_gradients:
                #     progress_bar.update(1)
                #     completed_steps += 1

                # if isinstance(checkpointing_steps, int):
                #     if completed_steps % checkpointing_steps == 0:
                #         output_dir = f"step_{completed_steps}"
                #         if args.output_dir is not None:
                #             output_dir = os.path.join(args.output_dir, output_dir)
                #         accelerator.save_state(output_dir)
                if completed_steps >= args.max_train_steps:
                    break
        lr_scheduler.step()

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            ### to cuda
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            # losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
            losses.append(loss)

        filename = args.checkpoint_path + '/{}_{}_{}_{}_{}_full.pth.tar'.format(args.model, args.dataset,
                                                                                args.compressor,
                                                                                args.compressor_ratio, epoch)
        train_end = time.time()
        train_dur = train_end - train_begin
        print("| epoch {} cost {} training time |".format(epoch, train_dur))

        if hvd.rank() == 0:
            torch.save({
                'epoch': epoch + 1,
                'net': 'GPT2',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename)

        # 将losses中的零维标量转换为1维张量
        losses = [loss.unsqueeze(0) for loss in losses]
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")


def load_base_checkpoint(model, optimizer, args):
    if args.resume is not None and args.resume != '':
        filepath = args.resume_path + args.model + '_' + args.dataset + '_' + args.compressor + '_' + str(
            args.compressor_ratio) + '_' + str(args.resume) + '_full.pth.tar'
        print("resume from {}".format(filepath))
    if os.path.isfile(filepath):
        begin = time.time()
        print("loading {}".format(filepath))
        checkpoint = torch.load(filepath)
        args.start_epoch = checkpoint['epoch']
        print("loading model")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        end = time.time()
        print("loaded base checkpoint '{}' (epoch {}) cost {}".format(args.resume, checkpoint['epoch'], end - begin))
        return model, optimizer
    else:
        raise ValueError("No checkpoint found")


def load_differential_checkpoint(model, optimizer, train_loader_len, topk, args):
    train_loader_len = args.fullfreq
    named_parameters = list(model.named_parameters())
    _parameter_names = {v: k for k, v in sorted(named_parameters)}
    checkpoint_index_parts = args.resume.split('-')
    for i in range(0, train_loader_len):
        # TODO:Reset optimizer gradient
        optimizer.zero_grad()
        second_part = int(checkpoint_index_parts[1]) + i

        filepath = (args.resume_path + args.model + '_' + args.dataset + '_' + args.compressor + '_' +
                    str(args.compressor_ratio) + '_' + str(checkpoint_index_parts[0])
                    + '-' + str(second_part) + '_diff.pth.tar')
        tensor_compressed = torch.load(filepath)

        for key in tensor_compressed.keys():
            tensor = topk.decompress(tensor_compressed[key]['tensors'], tensor_compressed[key]['ctx'], None)
            for param_group in optimizer.param_groups:
                for p in param_group['params']:
                    name = _parameter_names.get(p)
                    if name == key:
                        p.grad = tensor
                        break
        optimizer.step()
        print("loaded differential checkpoint {}".format(str(checkpoint_index_parts[0])
                                                         + '-' + str(second_part)))

    return model, optimizer


def new_load_differential_checkpoint(model, optimizer, topk, args):
    named_parameters = list(model.named_parameters())
    _parameter_names = {v: k for k, v in sorted(named_parameters)}

    thread_base = threading.Thread(target=base_plus_differential, args=(model, optimizer, _parameter_names, topk, args))
    thread_diff = threading.Thread(target=differential_plus_differential,
                                   args=(model, optimizer, _parameter_names, topk, args))

    begin = time.time()
    thread_base.start()
    thread_diff.start()

    thread_base.join()
    thread_diff.join()
    # with multiprocessing.Pool(processes=2) as pool:
    #     result_base = pool.apply_async(base_plus_differential, args=(model, optimizer, _parameter_names, topk))
    #     result_diff = pool.apply_async(differential_plus_differential, args=(model, optimizer, _parameter_names, topk))
    #     result_d = result_diff.get()
    #     result_b = result_base.get()
    #     end = time.time()
    #     print("Multiprocessing cost {} time".format(end - begin))

    begin_temp = time.time()
    optimizer.zero_grad()
    # Attention: need to update optimizer's grad
    for name, tensor in gradient_accumulation_group.items():
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if name == _parameter_names.get(p):
                    p.grad = tensor
                    break

    optimizer.step()
    end = time.time()
    print("after async optimizer step cost {} time".format(end - begin_temp))
    print("new_load_differential_checkpoint Async Loading totally cost {} time".format(end - begin))

    return model, optimizer
    # for i in range(0, accum_len):
    #     optimizer.zero_grad()
    #     filepath = (filedir + args.model + '_' + args.dataset + '_' + args.compressor + '_' +
    #                 str(args.compressor_ratio) + '_' + str(args.resume) + '-' + str(i) + '_diff.pth.tar')
    #     tensor_compressed = torch.load(filepath)
    #
    #     for key in tensor_compressed.keys():
    #         tensor = topk.decompress(tensor_compressed[key]['tensors'], tensor_compressed[key]['ctx'], None)
    #         for param_group in optimizer.param_groups:
    #             for p in param_group['params']:
    #                 name = _parameter_names.get(p)
    #                 if name == key:
    #                     p.grad = tensor
    #                     break
    #     optimizer.step()
    #     print("loaded iteration {}".format(i))


def base_plus_differential(model, optimizer, _parameter_names, topk, args):
    begin = time.time()
    optimizer.zero_grad()
    checkpoint_index_parts = args.resume.split('-')
    second_part = int(checkpoint_index_parts[1]) + 1
    filepath = (args.resume_path + args.model + '_' + args.dataset + '_' + args.compressor + '_' +
                str(args.compressor_ratio) + '_' + str(checkpoint_index_parts[0]) + '-' + str(
                second_part) + '_diff.pth.tar')
    tensor_compressed = torch.load(filepath)

    for key in tensor_compressed.keys():
        tensor = topk.decompress(tensor_compressed[key]['tensors'], tensor_compressed[key]['ctx'], None)
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                name = _parameter_names.get(p)
                if name == key:
                    p.grad = tensor
                    break
    optimizer.step()
    end = time.time()
    cost_time = end - begin
    print("loaded single differential checkpoint with base checkpoint cost {} time".format(cost_time))


def differential_plus_differential(model, optimizer, _parameter_names, topk, args):
    begin = time.time()
    global gradient_accumulation_group
    checkpoint_index_parts = args.resume.split('-')
    for i in range(2, 10):
        second_part = int(checkpoint_index_parts[1]) + i
        filepath = (args.resume_path + args.model + '_' + args.dataset + '_' + args.compressor + '_' +
                    str(args.compressor_ratio) + '_' + str(checkpoint_index_parts[0]) + '-' + str(
                    second_part) + '_diff.pth.tar')
        tensor_compressed = torch.load(filepath)
        for key in tensor_compressed.keys():
            tensor = topk.decompress(tensor_compressed[key]['tensors'], tensor_compressed[key]['ctx'], None)
            for param_group in optimizer.param_groups:
                for p in param_group['params']:
                    name = _parameter_names.get(p)
                    if name == key:
                        if name not in gradient_accumulation_group:
                            gradient_accumulation_group[name] = tensor.detach().clone()
                        else:
                            gradient_accumulation_group[name].add_(tensor)
                        break

    end = time.time()
    cost_time = end - begin
    print("loaded eight differential checkpoints with differential checkpoints cost {} time".format(cost_time))


def create_checkpoint_dir(checkpoint_dir, args):
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")

    # 组合文件夹名
    folder_name = f"{args.model}_{args.dataset}_{args.compressor_ratio}_{time_str}"
    full_path = os.path.join(checkpoint_dir, folder_name)

    # 创建文件夹
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path


if __name__ == "__main__":
    main()
