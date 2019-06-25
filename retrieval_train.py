#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from empchat.datasets.loader import TrainEnvironment
from empchat.models import (
    create as create_model,
    load as load_model,
    load_embeddings,
    save as save_model,
    score_candidates,
)
from empchat.util import get_logger, get_opt


def loss_fn(ctx, labels):
    assert (
        ctx.size() == labels.size()
    ), f"ctx.size : {ctx.size()}, labels.size: {labels.size()}"
    # both are [batch, dim]
    batch_size = ctx.size(0)
    dot_products = ctx.mm(labels.t())
    # [batch, batch]
    log_prob = F.log_softmax(dot_products, dim=1)
    targets = log_prob.new_empty(batch_size).long()
    targets = torch.arange(batch_size, out=targets)
    loss = F.nll_loss(log_prob, targets)
    nb_ok = (log_prob.max(dim=1)[1] == targets).float().sum()
    return loss, nb_ok


def train(epoch, start_time, model, optimizer, opt_, data_loader):
    """Run through one epoch of model training with the provided data loader."""
    model.train()
    # Initialize meters + timers
    train_loss = 0
    nb_ok = 0
    nb_exs = 0
    nb_losses = 0
    epoch_start = time.time()
    # Run one epoch
    for idx, ex in enumerate(data_loader, 1):
        params = [
            field.cuda(non_blocking=True)
            if opt_.cuda
            else field
            if field is not None
            else None
            for field in ex
        ]
        loss, ok = loss_fn(*model(*params))
        nb_ok += ok
        nb_exs += ex[0].size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.sum().item()
        nb_losses += 1
        if idx % opt_.display_iter == 0 or idx == len(data_loader):
            avg_loss = train_loss / nb_losses
            acc = 100 * nb_ok / nb_exs
            elapsed = time.time() - start_time
            logging.info(
                f"train: Epoch = {epoch} | iter = {idx}/{len(data_loader)} | loss = "
                f"{avg_loss:.3f} | batch P@1 = {acc:.2f} % | elapsed time = "
                f"{elapsed:.2f} (s)"
            )
            train_loss = 0
            nb_losses = 0
    epoch_elapsed = time.time() - epoch_start
    logging.info(
        f"train: Epoch {epoch:d} done. Time for epoch = {epoch_elapsed:.2f} (s)"
    )


def validate(
    epoch,
    model,
    data_loader,
    max_exs=100000,
    is_test=False,
    nb_candidates=100,
    shuffled_str="shuffled",
):
    model.eval()
    examples = 0
    eval_start = time.time()
    sum_losses = 0
    n_losses = 0
    correct = 0
    all_context = []
    all_cands = []
    n_skipped = 0
    dtype = model.module.opt.dataset_name
    for i, ex in enumerate(data_loader):
        batch_size = ex[0].size(0)
        if dtype == "reddit" and is_test and n_skipped < max_exs:
            n_skipped += batch_size
            continue
        params = [
            field.cuda(non_blocking=True)
            if opt.cuda
            else field
            if field is not None
            else None
            for field in ex
        ]
        ctx, cands = model(*params)
        all_context.append(ctx)
        all_cands.append(cands)
        loss, nb_ok = loss_fn(ctx, cands)
        sum_losses += loss
        correct += nb_ok
        n_losses += 1
        examples += batch_size
        if examples >= max_exs and dtype == "reddit":
            break
    n_examples = 0
    if len(all_context) > 0:
        logging.info("Processing candidate top-K")
        all_context = torch.cat(all_context, dim=0)  # [:50000]  # [N, 2h]
        all_cands = torch.cat(all_cands, dim=0)  # [:50000]  # [N, 2h]
        acc_ranges = [1, 3, 10]
        n_correct = {r: 0 for r in acc_ranges}
        for context, cands in list(
            zip(all_context.split(nb_candidates), all_cands.split(nb_candidates))
        )[:-1]:
            _, top_answers = score_candidates(context, cands)
            n_cands = cands.size(0)
            gt_index = torch.arange(n_cands, out=top_answers.new(n_cands, 1))
            for acc_range in acc_ranges:
                n_acc = (top_answers[:, :acc_range] == gt_index).float().sum()
                n_correct[acc_range] += n_acc
            n_examples += n_cands
        accuracies = {r: 100 * n_acc / n_examples for r, n_acc in n_correct.items()}
        avg_loss = sum_losses / (n_losses + 0.00001)
        avg_acc = 100 * correct / (examples + 0.000001)
        valid_time = time.time() - eval_start
        logging.info(
            f"Valid ({shuffled_str}): Epoch = {epoch:d} | avg loss = {avg_loss:.3f} | "
            f"batch P@1 = {avg_acc:.2f} % | "
            + f" | ".join(
                f"P@{k},{nb_candidates} = {v:.2f}%" for k, v in accuracies.items()
            )
            + f" | valid time = {valid_time:.2f} (s)"
        )
        return avg_loss
    return 10


def train_model(opt_):
    env = TrainEnvironment(opt_)
    dictionary = env.dict
    if opt_.load_checkpoint:
        net, dictionary = load_model(opt_.load_checkpoint, opt_)
        env = TrainEnvironment(opt_, dictionary)
        env.dict = dictionary
    else:
        net = create_model(opt_, dictionary["words"])
        if opt_.embeddings and opt_.embeddings != "None":
            load_embeddings(opt_, dictionary["words"], net)
    paramnum = 0
    trainable = 0
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            trainable += parameter.numel()
        paramnum += parameter.numel()
    print("TRAINABLE", paramnum, trainable)
    if opt_.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    if opt_.optimizer == "adamax":
        lr = opt_.learning_rate or 0.002
        named_params_to_optimize = filter(
            lambda p: p[1].requires_grad, net.named_parameters()
        )
        params_to_optimize = (p[1] for p in named_params_to_optimize)
        optimizer = optim.Adamax(params_to_optimize, lr=lr)
        if opt_.epoch_start != 0:
            saved_params = torch.load(
                opt_.load_checkpoint, map_location=lambda storage, loc: storage
            )
            optimizer.load_state_dict(saved_params["optim_dict"])
    else:
        lr = opt_.learning_rate or 0.01
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr
        )
    start_time = time.time()
    best_loss = float("+inf")
    test_data_shuffled = env.build_valid_dataloader(True)
    test_data_not_shuffled = env.build_valid_dataloader(False)
    with torch.no_grad():
        validate(
            0,
            net,
            test_data_shuffled,
            nb_candidates=opt_.hits_at_nb_cands,
            shuffled_str="shuffled",
        )
    train_data = None
    for epoch in range(opt_.epoch_start, opt_.num_epochs):
        if train_data is None or opt_.dataset_name == "reddit":
            train_data = env.build_train_dataloader(epoch)
        train(epoch, start_time, net, optimizer, opt_, train_data)
        with torch.no_grad():
            # We compute the loss both for shuffled and not shuffled case.
            # however, the loss that determines if the model is better is the
            # same as the one used for training.
            loss_shuffled = validate(
                epoch,
                net,
                test_data_shuffled,
                nb_candidates=opt_.hits_at_nb_cands,
                shuffled_str="shuffled",
            )
            loss_not_shuffled = validate(
                epoch,
                net,
                test_data_not_shuffled,
                nb_candidates=opt_.hits_at_nb_cands,
                shuffled_str="not-shuffled",
            )
            if opt_.no_shuffle:
                loss = loss_not_shuffled
            else:
                loss = loss_shuffled
            if loss < best_loss:
                best_loss = loss
                best_loss_epoch = epoch
                logging.info(f"New best loss, saving model to {opt_.model_file}")
                save_model(opt_.model_file, net, dictionary, optimizer)
            # Stop if it's been too many epochs since the loss has decreased
            if opt_.stop_crit_num_epochs != -1:
                if epoch - best_loss_epoch >= opt_.stop_crit_num_epochs:
                    break
    return net, dictionary


def main(opt_):
    if opt_.pretrained:
        net, dictionary = load_model(opt_.pretrained, opt_)
        net.opt.dataset_name = opt_.dataset_name
        net.opt.reddit_folder = opt_.reddit_folder
        net.opt.reactonly = opt_.reactonly
        net.opt.max_hist_len = opt_.max_hist_len
        env = TrainEnvironment(net.opt, dictionary)
        if opt_.cuda:
            net = torch.nn.DataParallel(net.cuda())
        valid_data = env.build_valid_dataloader(False)
        test_data = env.build_valid_dataloader(False, test=True)
        with torch.no_grad():
            logging.info("Validating on the valid set -unshuffled")
            validate(
                0, net, valid_data, is_test=False, nb_candidates=opt_.hits_at_nb_cands
            )
            logging.info("Validating on the hidden test set -unshuffled")
            validate(
                0, net, test_data, is_test=True, nb_candidates=opt_.hits_at_nb_cands
            )
        valid_data = env.build_valid_dataloader(True)
        test_data = env.build_valid_dataloader(True, test=True)
        with torch.no_grad():
            logging.info("Validating on the valid set -shuffle")
            validate(
                0, net, valid_data, is_test=False, nb_candidates=opt_.hits_at_nb_cands
            )
            logging.info("Validating on the hidden test set -shuffle")
            validate(
                0, net, test_data, is_test=True, nb_candidates=opt_.hits_at_nb_cands
            )
    else:
        train_model(opt_)


if __name__ == "__main__":
    opt = get_opt()
    # Set random state
    torch.manual_seed(opt.random_seed)
    opt.cuda = opt.cuda and torch.cuda.is_available()
    if opt.cuda:
        torch.cuda.manual_seed(opt.random_seed)
    # Set logging
    logger = get_logger(opt)
    main(opt)
