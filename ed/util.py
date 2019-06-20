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


def get_opt(empty=False):
    parser = get_parser()
    opt = parser.parse_args() if not empty else parser.parse_args([])
    set_defaults(opt)
    return opt


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-add-transformer-layer", action="store_true", help="Add final Transformer layer to BERT model")
    parser.add_argument("--bert-dim", type=int, default=512, help="Final BERT Transformer layer output dim")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--dailydialog-folder", type=str, help="Path to DailyDialog data folder")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="reddit",
        choices=["reddit", "empchat", "dailydialog"],
        help='Data to train/eval on'
    )
    parser.add_argument("--dict-max-words", type=int, default=250000, help='Max dictionary size (not used with BERT)')
    parser.add_argument("--display-iter", type=int, default=250, help='Frequency of train logging')
    parser.add_argument("--embeddings", type=str)
    parser.add_argument("--embeddings-size", type=int, default=300)
    parser.add_argument("--empchat-folder", type=str)
    parser.add_argument("--fast-debug", action="store_true")
    parser.add_argument("--fix-mean", action="store_true", default=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--hits-at-nb-cands", type=int, default=100)
    parser.add_argument("--learn-embeddings", action="store_true")
    parser.add_argument("--load-checkpoint", type=str)
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--max-hist-len", type=int, default=1)
    parser.add_argument("--max-sent-len", type=int, default=100)
    parser.add_argument("--rm-long-sent", action="store_true")
    parser.add_argument("--rm-long-contexts", action="store_true")
    parser.add_argument("--model", type=str, choices=["bert", "transformer"])
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--normalize-emb", action="store_true")
    parser.add_argument("--normalize-sent-emb", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--opt-additional-only", action="store_true")
    parser.add_argument(
        "--optimizer", type=str, choices=["sgd", "adamax"], default="adamax"
    )
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--random-seed", type=int, default=92179)
    parser.add_argument("--reddit-folder", type=str)
    parser.add_argument("--stop-crit-num-epochs", type=int, default=-1)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument("--transformer-dropout", type=float, default=0)
    parser.add_argument("--transformer-n-heads", type=int, default=8)
    parser.add_argument("--transformer-response-hmul", type=int, default=1)
    parser.add_argument("--two-transformers", action="store_true")
    parser.add_argument("--use-manual-norm", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=32)
    parser.add_argument("-e", "--epoch-start", type=int, default=0)
    parser.add_argument("-lr", "--learning-rate", type=float, default=None)
    parser.add_argument("--reactonly", action="store_true")
    parser.add_argument("--emp-loss", type=str, default=None)
    parser.add_argument("--fasttext", type=int, default=None)
    parser.add_argument("--fasttext-path", type=str, default=None)
    parser.add_argument("--fasttext-type", type=str, default=None)
    return parser


def set_defaults(opt):
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
