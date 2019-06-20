# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import json
import logging
import os
import sys


def get_opt(existing_opt=None):
    parser = get_parser()
    opt = parser.parse_args([]) if existing_opt is not None else parser.parse_args()
    # If we have an existing set of options, just use defaults for our new set. We'll
    # transfer over needed existing option values below
    set_defaults(opt=opt, existing_opt=existing_opt)
    return opt


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Training/eval batch size"
    )
    parser.add_argument(
        "--bert-add-transformer-layer",
        action="store_true",
        help="Add final Transformer layer to BERT model",
    )
    parser.add_argument(
        "--bert-dim",
        type=int,
        default=512,
        help="Final BERT Transformer layer output dim",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument(
        "--dailydialog-folder", type=str, help="Path to DailyDialog data folder"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="reddit",
        choices=["reddit", "empchat", "dailydialog"],
        help="Data to train/eval on",
    )
    parser.add_argument(
        "--dict-max-words",
        type=int,
        default=250000,
        help="Max dictionary size (not used with BERT)",
    )
    parser.add_argument(
        "--display-iter", type=int, default=250, help="Frequency of train logging"
    )
    parser.add_argument("--embeddings", type=str, help="Path to embeddings file")
    parser.add_argument(
        "--embeddings-size", type=int, default=300, help="Transformer embedding size"
    )
    parser.add_argument(
        "--empchat-folder", type=str, help="Path to EmpatheticDialogues data folder"
    )
    parser.add_argument(
        "-e",
        "--epoch-start",
        type=int,
        default=0,
        help="Initial epoch number when resuming training",
    )
    parser.add_argument(
        "--fasttext",
        type=int,
        default=None,
        help="Number of fastText labels to prepend",
    )
    parser.add_argument(
        "--fasttext-path", type=str, default=None, help="Path to fastText classifier"
    )
    parser.add_argument(
        "--fasttext-type",
        type=str,
        default=None,
        help="Specifies labels of fastText classifier",
    )
    parser.add_argument(
        "--hits-at-nb-cands",
        type=int,
        default=100,
        help="Num candidates to calculate precision out of",
    )
    parser.add_argument(
        "--learn-embeddings", action="store_true", help="Train on embeddings"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=None,
        help="Training learning rate",
    )
    parser.add_argument("--load-checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    parser.add_argument(
        "--max-hist-len",
        type=int,
        default=1,
        help="Max num conversation turns to use in context",
    )
    parser.add_argument(
        "--max-sent-len", type=int, default=100, help="Max num tokens per sentence"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "transformer"],
        help="Choice of retrieval model",
    )
    parser.add_argument("--model-dir", type=str, help="Model save folder")
    parser.add_argument("--model-name", type=str, help="Model save name")
    parser.add_argument(
        "--n-layers", type=int, default=6, help="Num Transformer layers"
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Don't shuffle data during training"
    )
    parser.add_argument(
        "--normalize-emb", action="store_true", help="Normalize loaded embeddings"
    )
    parser.add_argument(
        "--normalize-sent-emb",
        action="store_true",
        help="Normalize context/candidate embeddings",
    )
    parser.add_argument("--num-epochs", type=int, default=1000, help="Num epochs")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adamax"],
        default="adamax",
        help="Choice of optimizer",
    )
    parser.add_argument(
        "--pretrained", type=str, help="Path to pretrained model (will run eval only)"
    )
    parser.add_argument("--random-seed", type=int, default=92179, help="Seed")
    parser.add_argument(
        "--reactonly",
        action="store_true",
        help="EmpatheticDialogues: only consider Listener responses",
    )
    parser.add_argument("--reddit-folder", type=str, help="Path to Reddit data folder")
    parser.add_argument(
        "--rm-long-sent",
        action="store_true",
        help="Completely remove long Reddit sentences",
    )
    parser.add_argument(
        "--rm-long-contexts",
        action="store_true",
        help="Completely remove long Reddit contexts",
    )
    parser.add_argument(
        "--stop-crit-num-epochs",
        type=int,
        default=-1,
        help="Num epochs to stop after if loss is not decreasing",
    )
    parser.add_argument(
        "--transformer-dim",
        type=int,
        default=512,
        help="Input Transformer embedding dim",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0,
        help="Transformer attention/FFN dropout",
    )
    parser.add_argument(
        "--transformer-n-heads",
        type=int,
        default=8,
        help="Num Transformer attention heads",
    )
    return parser


def set_defaults(opt, existing_opt=None):
    if opt.model_dir is None:
        # retrieval_eval_bleu.py uses an `output_folder` arg instead
        assert existing_opt.output_folder is not None
        opt.model_dir = existing_opt.output_folder

    # Set model directory
    os.makedirs(opt.model_dir, exist_ok=True)

    # Set model name
    if not opt.model_name:
        import uuid
        import time

        opt.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    if opt.log_file is None:
        opt.log_file = os.path.join(opt.model_dir, opt.model_name + ".txt")
    opt.model_file = os.path.join(opt.model_dir, opt.model_name + ".mdl")


def get_logger(opt):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if opt.log_file:
        logfile = logging.FileHandler(opt.log_file, "a")
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    command = " ".join(sys.argv)
    logger.info(f"COMMAND: {command}")
    logger.info("-" * 100)
    config = json.dumps(vars(opt), indent=4, sort_keys=True)
    logger.info(f"CONFIG:\n{config}")
    return logger
