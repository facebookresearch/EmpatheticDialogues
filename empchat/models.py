# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import torch
import torch.nn as nn

from empchat.bert_local import BertAdapter
from empchat.datasets.tokens import PAD_TOKEN
from empchat.transformer_local import TransformerAdapter


def load_embeddings(opt, dictionary, model):
    path = opt.embeddings
    logging.info(f"Loading embeddings file from {path}")
    emb_table = model.embeddings.weight
    requires_grad = emb_table.requires_grad
    emb_table[dictionary[PAD_TOKEN]].zero_()  # Zero-out padding index
    n_added = 0
    missing_dict = set(dictionary.keys())
    with open(path) as f:
        for line in f:
            parsed = line.rstrip().split(" ")
            assert len(parsed) == opt.embeddings_size + 1
            w = parsed[0]
            if w in dictionary:
                n_added += 1
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                if opt.normalize_emb:
                    vec = vec / vec.norm(2)
                emb_table.data[dictionary[w]] = vec
                missing_dict.remove(w)
    sample = ", ".join(list(missing_dict)[:8])
    logging.info(
        f"Loaded {n_added} vectors from embeddings file; {len(missing_dict)} are "
        f"missing, among which: {sample}"
    )
    emb_table.detach_()
    emb_table.requires_grad = requires_grad


def save(filename, net, dictionary, optimizer):
    if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        net = net.module
    state_dict = net.state_dict()
    params = {
        "state_dict": state_dict,
        "word_dict": dictionary,
        "opt": net.opt,
        "optim_dict": optimizer.state_dict(),
    }
    torch.save(params, filename)


def load(filename, new_opt):
    logging.info(f"Loading model {filename}")
    saved_params = torch.load(filename, map_location=lambda storage, loc: storage)
    word_dict = saved_params["word_dict"]
    state_dict = saved_params["state_dict"]
    saved_opt = saved_params["opt"]
    for k, v in vars(new_opt).items():
        if not hasattr(saved_opt, k):
            logging.warning(f"Setting {k} to {v}")
            setattr(saved_opt, k, v)
    if not (hasattr(new_opt, "fasttext")):
        setattr(saved_opt, "fasttext", new_opt.fasttext)
    if new_opt.model == "bert":
        assert "bert_tokenizer" in word_dict
    net = create(saved_opt, word_dict["words"])
    net.load_state_dict(state_dict, strict=False)
    return net, word_dict


def create(opt, dict_words):
    if opt.model == "bert":
        return BertAdapter(opt, dict_words)
    elif opt.model == "transformer":
        return TransformerAdapter(opt, dict_words)
    else:
        raise ValueError("Model not recognized!")


def score_candidates(all_context, all_cands, top_k=20, normalize=False):
    # all_context is of size [ctx, d]
    # all_cands is of size [cand, d]
    dot_products = all_context.mm(all_cands.t())  # [ctx, cand]
    if normalize:
        dot_products /= all_context.norm(2, dim=1).unsqueeze(1)
        dot_products /= all_cands.norm(2, dim=1).unsqueeze(0)
    scores, answers = dot_products.topk(top_k, dim=1)
    # Index of top-k items in decreasing order. Answers is of size [ctx, top_k]
    return scores, answers
