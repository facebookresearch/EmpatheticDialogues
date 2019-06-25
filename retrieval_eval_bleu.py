#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import os

import torch
from fairseq import bleu
from tqdm import tqdm

from empchat.datasets.dailydialog import DDDataset
from empchat.datasets.empchat import EmpDataset
from empchat.datasets.reddit import RedditDataset
from empchat.datasets.parlai_dictionary import ParlAIDictionary
from empchat.datasets.tokens import tokenize, PAD_TOKEN, START_OF_COMMENT, UNK_TOKEN
from empchat.models import load as load_model, score_candidates
from empchat.util import get_opt


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bleu-dict",
    type=str,
    default=None,
    help=(
        "Path to dictionary to use for BLEU calculation (if "
        "not the same as the dictionary to use for retrieval)"
    ),
)
parser.add_argument(
    "--candidates", type=str, default=None, help="Path to candidates to use"
)
parser.add_argument(
    "--dailydialog-cands", action="store_true", help="Include DailyDialog candidates"
)
parser.add_argument(
    "--dailydialog-folder", type=str, help="Path to DailyDialog data folder"
)
parser.add_argument(
    "--empchat-cands",
    action="store_true",
    help="Include EmpatheticDialogues candidates",
)
parser.add_argument(
    "--empchat-folder", type=str, help="Path to EmpatheticDialogues data folder"
)
parser.add_argument(
    "--fasttext", type=int, default=None, help="Number of fastText labels to prepend"
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
parser.add_argument("--gpu", type=int, default=-1, help="Specify GPU device id to use")
parser.add_argument(
    "--max-cand-length",
    type=int,
    default=20,
    help="Max candidate length in number of tokens",
)
parser.add_argument(
    "--max-hist-len",
    type=int,
    default=1,
    help="Max num conversation turns to use in context",
)
parser.add_argument(
    "--model", "--pretrained", type=str, default=None, help="Path to model to use"
)
parser.add_argument(
    "--n-candidates", type=int, default=int(1e6), help="Max number of candidates"
)
parser.add_argument("--name", type=str, help="Part of name of response output file")
parser.add_argument("--no-cuda", action="store_true", help="Use CPU only")
parser.add_argument(
    "--normalize-cands", action="store_true", help="Normalize encoded candidates"
)
parser.add_argument(
    "--output-folder", type=str, default=None, help="Path to output folder"
)
parser.add_argument(
    "--reactonly",
    action="store_true",
    help="EmpatheticDialogues: only consider Listener responses",
)
parser.add_argument(
    "--reddit-cands", action="store_true", help="Include Reddit candidates"
)
parser.add_argument("--reddit-folder", type=str, help="Path to Reddit data folder")
parser.add_argument(
    "--save-candidates", action="store_true", help="If true, save candidate files"
)
parser.add_argument(
    "--task",
    type=str,
    choices=["dailydialog", "empchat", "reddit"],
    default="empchat",
    help="Dataset for context/target-response pairs",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info(f"CUDA enabled (GPU {args.gpu:d})")
else:
    logger.info("Running on CPU only.")
if args.fasttext is not None:
    args.max_cand_length += args.fasttext
net, net_dictionary = load_model(args.model, get_opt(existing_opt=args))
if "bert_tokenizer" in net_dictionary:
    if args.task == "dailydialog":
        raise NotImplementedError("BERT model currently incompatible with DailyDialog!")
if args.bleu_dict is not None:
    _, bleu_dictionary = load_model(args.bleu_dict, get_opt(existing_opt=args))
else:
    bleu_dictionary = net_dictionary
paramnum = 0
trainable = 0
for parameter in net.parameters():
    if parameter.requires_grad:
        trainable += parameter.numel()
    paramnum += parameter.numel()
print(paramnum, trainable)
print(type(net_dictionary))
NET_PAD_IDX = net_dictionary["words"][PAD_TOKEN]
NET_UNK_IDX = net_dictionary["words"][UNK_TOKEN]
print(type(bleu_dictionary))
BLEU_PAD_IDX = bleu_dictionary["words"][PAD_TOKEN]
BLEU_UNK_IDX = bleu_dictionary["words"][UNK_TOKEN]
BLEU_EOS_IDX = bleu_dictionary["words"][START_OF_COMMENT]
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info(f"CUDA enabled (GPU {args.gpu:d})")
else:
    logger.info("Running on CPU only.")
actual_ct = [0, 0, 0]
if args.cuda:
    net = torch.nn.DataParallel(net)
    net.cuda()
net.eval()


def pad(items):
    max_len = max(len(i) for i in items)
    tensor = torch.LongTensor(len(items), max_len).fill_(NET_PAD_IDX)
    for i, sentence in enumerate(items):
        tensor[i, : sentence.size(0)] = sentence
    return tensor


def build_candidates(
    max_cand_length, n_cands=int(1e7), rm_duplicates=True, rm_starting_gt=True
):
    global actual_ct
    global args
    tensor = torch.LongTensor(n_cands, max_cand_length).fill_(NET_PAD_IDX)
    i = 0
    chunk = 422
    if "bert_tokenizer" in net_dictionary:
        gt_tokens = torch.LongTensor(
            net_dictionary["bert_tokenizer"].convert_tokens_to_ids(["&", "g", "##t"])
        )
    else:
        gt_index = net_dictionary["words"]["&gt"]
        lt_index = net_dictionary["words"]["&lt"]
    unk_index = net_dictionary["words"]["<UNK>"]
    n_duplicates = n_start_gt = 0
    if rm_duplicates:
        all_sent = set()

    def _has_lts(sentence_) -> bool:
        if "bert_tokenizer" in net_dictionary:
            tokens = net_dictionary["bert_tokenizer"].convert_ids_to_tokens(
                sentence_.tolist()
            )
            return "& l ##t" in " ".join(tokens)
        else:
            return torch.sum(sentence_ == lt_index).gt(0)

    def _starts_with_gt(sentence_) -> bool:
        if "bert_tokenizer" in net_dictionary:
            if sentence_.size(0) < 3:
                return False
            else:
                return torch.eq(sentence_[:3], gt_tokens).all()
        else:
            return sentence_[0].item == gt_index

    parlai_dict = ParlAIDictionary.create_from_reddit_style(net_dictionary)
    if args.empchat_cands:
        dataset = EmpDataset(
            "train",
            parlai_dict,
            data_folder=args.empchat_folder,
            reactonly=False,
            fasttext=args.fasttext,
            fasttext_type=args.fasttext_type,
            fasttext_path=args.fasttext_path,
        )
        sample_index = range(len(dataset))
        for data_idx in sample_index:
            _context, sentence, _ = dataset[data_idx]
            sent_length = sentence.size(0)
            if torch.sum(sentence == unk_index).gt(0):
                continue
            if _has_lts(sentence):
                continue
            if sent_length <= max_cand_length:
                if _starts_with_gt(sentence) and rm_starting_gt:
                    n_start_gt += 1
                    continue
                if rm_duplicates:
                    tuple_sent = tuple(sentence.numpy())
                    if tuple_sent in all_sent:
                        n_duplicates += 1
                        continue
                    all_sent.add(tuple_sent)
                tensor[i, : sentence.size(0)] = sentence
                i += 1
                if i >= n_cands:
                    break
    breakpoint_ = i
    actual_ct[1] = i
    if args.dailydialog_cands:
        dataset = DDDataset("train", parlai_dict, data_folder=args.dailydialog_folder)
        sample_index = range(len(dataset))
        for data_idx in sample_index:
            _context, sentence = dataset[data_idx]
            sent_length = sentence.size(0)
            if torch.sum(sentence == unk_index).gt(0):
                continue
            if _has_lts(sentence):
                continue
            if sent_length <= max_cand_length:
                if _starts_with_gt(sentence) and rm_starting_gt:
                    n_start_gt += 1
                    continue
                if rm_duplicates:
                    tuple_sent = tuple(sentence.numpy())
                    if tuple_sent in all_sent:
                        n_duplicates += 1
                        continue
                    all_sent.add(tuple_sent)
                tensor[i, : sentence.size(0)] = sentence
                i += 1
                if i >= n_cands:
                    break
    bp2 = i
    actual_ct[2] = i - breakpoint_
    if args.reddit_cands:
        while i < n_cands:
            chunk += 1
            logging.info(f"Loaded {i} / {n_cands} candidates")
            dataset = RedditDataset(args.reddit_folder, chunk, net_dictionary)
            sample_index = range(len(dataset))
            for data_idx in sample_index:
                _context, sentence = dataset[data_idx]
                sent_length = sentence.size(0)
                if sent_length == 0:
                    print(f"Reddit sentence {data_idx} is of length 0.")
                    continue
                if torch.sum(sentence == unk_index).gt(0):
                    continue
                if _has_lts(sentence):
                    continue
                if sent_length <= max_cand_length:
                    if _starts_with_gt(sentence) and rm_starting_gt:
                        n_start_gt += 1
                        continue
                    if rm_duplicates:
                        tuple_sent = tuple(sentence.numpy())
                        if tuple_sent in all_sent:
                            n_duplicates += 1
                            continue
                        all_sent.add(tuple_sent)
                    tensor[i, : sentence.size(0)] = sentence
                    i += 1
                    if i >= n_cands:
                        break
    actual_ct[0] = i - bp2
    logging.info(
        f"Loaded {i} candidates, {n_start_gt} start with >, {n_duplicates} duplicates"
    )
    args.n_candidates = i
    return tensor[:i, :], breakpoint_, bp2


def embed_candidates(candidates):
    out_tensor = None
    i = 0
    ch = candidates.split(2048, dim=0)
    for chunk in tqdm(range(len(ch))):
        _, encoded_cand = net(None, ch[chunk])
        if out_tensor is None:
            out_tensor = torch.FloatTensor(candidates.size(0), encoded_cand.size(1))
            if args.cuda:
                out_tensor = out_tensor.cuda()
        if args.normalize_cands:
            encoded_cand /= encoded_cand.norm(2, dim=1, keepdim=True)
        batch_size = encoded_cand.size(0)
        out_tensor[i : i + batch_size] = encoded_cand
        i += batch_size
    return out_tensor


def get_token_tensor(sentence):
    words = net_dictionary["words"]
    tokenized = tokenize(sentence, split_sep=None)
    return torch.LongTensor([words.get(w, NET_UNK_IDX) for w in tokenized])


def stringify(tensor):
    iwords = net_dictionary["iwords"]
    assert tensor.squeeze().dim() == 1, "Wrong tensor size!"
    return " ".join(
        iwords[i] for i in tensor.squeeze().cpu().numpy() if i != NET_PAD_IDX
    ).replace(" ##", "")
    # Remove any BPE tokenization


if args.candidates:
    fixed_candidates = torch.load(args.candidates)
    if args.n_candidates < fixed_candidates.size(0):
        logging.warning(
            f"Keeping only {args.n_candidates} / {fixed_candidates.size(0)} candidates"
        )
        fixed_candidates = fixed_candidates[: args.n_candidates]
else:
    fixed_candidates, breakingpt, breakingpt2 = build_candidates(
        args.max_cand_length, args.n_candidates
    )
if args.cuda:
    fixed_candidates = fixed_candidates.cuda(non_blocking=True)
logging.warning("Embedding candidates")
with torch.no_grad():
    cand_embs = embed_candidates(fixed_candidates)
logging.warning("Done with candidates")
if args.save_candidates:
    cand_path = os.path.join(args.output_folder, "reddit_cands_tokens.bin")
    logging.warning(f"Saving candidates in {cand_path}")
    torch.save(fixed_candidates, cand_path)
    emb_path = os.path.join(args.output_folder, "reddit_cands.bin")
    logging.warning(f"Saving candidate embs in {emb_path}")
    torch.save(cand_embs, emb_path)
    txt_path = os.path.join(args.output_folder, "reddit_cands.txt")
    logging.warning(f"Saving candidate texts in {txt_path}")
    with open(txt_path, "w") as f:
        for candidate in fixed_candidates:
            f.write(stringify(candidate))
            f.write("\n")
    logging.warning("Done saving files")

# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def predict(context, top_n=5, normalize=False):
    """
    returns a list of top_n tuples ("sentence", "score")
    """
    with torch.no_grad():
        context = context.unsqueeze(0)
        candidates = fixed_candidates
        if args.cuda:
            context = context.cuda(non_blocking=True)
        ctx, _ = net(context, None)
        scores, index = score_candidates(ctx, cand_embs, top_n, normalize)
        response = []
        outputs = []
        for i, (score, index) in enumerate(zip(scores.squeeze(0), index.squeeze(0)), 1):
            response.append((stringify(candidates[index]), float(score)))
            if index < breakingpt:
                outputs.append("EmpChat")
            elif index < breakingpt2:
                outputs.append("DailyDialog")
            else:
                outputs.append("Reddit")
        return response, outputs


def get_bleu4(split, history_len=1):
    """
    Print BLEU scores and output contexts and retrieved responses.
    """
    if history_len < 1:
        history_len = 1
    source_ct = [0, 0, 0]
    net_parlai_dict = ParlAIDictionary.create_from_reddit_style(net_dictionary)
    bleu_parlai_dict = ParlAIDictionary.create_from_reddit_style(bleu_dictionary)
    scorer = bleu.Scorer(BLEU_PAD_IDX, BLEU_EOS_IDX, BLEU_UNK_IDX)
    outf = open("retrieved_split_" + args.name + "_" + split + ".txt", "w")

    def _get_dataset(reddit_dict, parlai_dict):
        if args.task == "dailydialog":
            return DDDataset(
                split,
                parlai_dict,
                data_folder=args.dailydialog_folder,
                history_len=history_len,
            )
        elif args.task == "empchat":
            return EmpDataset(
                split,
                parlai_dict,
                data_folder=args.empchat_folder,
                history_len=history_len,
                reactonly=args.reactonly,
                fasttext=args.fasttext,
                fasttext_type=args.fasttext_type,
                fasttext_path=args.fasttext_path,
            )
        elif args.task == "reddit":
            return RedditDataset(
                data_folder=args.reddit_folder,
                chunk_id=999,
                dict_=reddit_dict,
                max_hist_len=history_len,
                rm_blank_sentences=True,
            )
        else:
            raise ValueError("Task unrecognized!")

    net_dataset = _get_dataset(net_dictionary, net_parlai_dict)
    bleu_dataset = _get_dataset(bleu_dictionary, bleu_parlai_dict)
    sample_index = range(len(bleu_dataset))
    for data_idx in sample_index:
        net_context, _ = net_dataset[data_idx][:2]
        bleu_context, bleu_sentence = bleu_dataset[data_idx][:2]
        target_tokens = bleu_sentence
        if args.fasttext is not None:
            target_tokens = target_tokens[args.fasttext :]
        context = bleu_parlai_dict.vec2txt(bleu_context.numpy().tolist())
        responses, sources = predict(net_context)
        response = responses[0][0]
        source = sources[0]
        if source == "Reddit":
            source_ct[0] += 1
        elif source == "EmpChat":
            source_ct[1] += 1
        else:
            source_ct[2] += 1
        if args.task == "empchat":
            cid, sid = bleu_dataset.getid(data_idx)
        else:
            cid = sid = -1
            # This is a hack, because the other datasets have no .getid() method
        if args.fasttext is not None:
            response = " ".join(response.split()[args.fasttext :])
        outf.write("\t".join([str(cid), str(sid), context, response, source]) + "\n")
        hypo_tokens = torch.IntTensor(bleu_parlai_dict.txt2vec(response))
        # Use this tokenization even if a BERT tokenizer exists, to match the BLEU
        # calculation when not using BERT
        scorer.add(target_tokens.type(torch.IntTensor), hypo_tokens)
    print(scorer.result_string(order=1))
    print(scorer.result_string(order=2))
    print(scorer.result_string(order=3))
    print(scorer.result_string(order=4))
    print(actual_ct)
    print(
        f"EmpatheticDialogues {int(source_ct[1]):d}: selected "
        f"{float(source_ct[1]) / sum(source_ct)}%, but total: "
        f"{float(actual_ct[1]) / sum(actual_ct)}"
    )
    print(
        f"DailyDialog {int(source_ct[2]):d}: selected "
        f"{float(source_ct[2]) / sum(source_ct)}%, but total: "
        f"{float(actual_ct[2]) / sum(actual_ct)}"
    )
    print(
        f"Reddit {int(source_ct[0]):d}: selected "
        f"{float(source_ct[0]) / sum(source_ct)}%, but total: "
        f"{float(actual_ct[0]) / sum(actual_ct)}"
    )


get_bleu4("valid", history_len=args.max_hist_len)
get_bleu4("test", history_len=args.max_hist_len)
