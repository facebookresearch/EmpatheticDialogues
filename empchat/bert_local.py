# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from empchat.datasets.tokens import (
    BERT_ID,
    EMPTYPERSONA_TOKEN,
    PAD_TOKEN,
    PARLAI_PAD_TOKEN,
    START_OF_COMMENT,
)


class BertAdapter(nn.Module):
    def __init__(self, opt, dictionary):
        from parlai.agents.bert_ranker.helpers import BertWrapper

        try:
            from pytorch_pretrained_bert import BertModel
        except ImportError:
            raise Exception(
                "BERT rankers needs pytorch-pretrained-BERT installed. "
                "\npip install pytorch-pretrained-bert"
            )
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[PAD_TOKEN]
        self.ctx_bert = BertWrapper(
            bert_model=BertModel.from_pretrained(BERT_ID),
            output_dim=opt.bert_dim,
            add_transformer_layer=opt.bert_add_transformer_layer,
        )
        self.cand_bert = BertWrapper(
            bert_model=BertModel.from_pretrained(BERT_ID),
            output_dim=opt.bert_dim,
            add_transformer_layer=opt.bert_add_transformer_layer,
        )

        # Reset the embeddings for the until-now unused BERT tokens
        orig_embedding_weights = BertModel.from_pretrained(
            BERT_ID
        ).embeddings.word_embeddings.weight
        mean_val = orig_embedding_weights.mean().item()
        std_val = orig_embedding_weights.std().item()
        unused_tokens = [START_OF_COMMENT, PARLAI_PAD_TOKEN, EMPTYPERSONA_TOKEN]
        unused_token_idxes = [dictionary[token] for token in unused_tokens]
        for token_idx in unused_token_idxes:
            rand_embedding = orig_embedding_weights.new_empty(
                (1, orig_embedding_weights.size(1))
            ).normal_(mean=mean_val, std=std_val)
            for embeddings in [
                self.ctx_bert.bert_model.embeddings.word_embeddings,
                self.cand_bert.bert_model.embeddings.word_embeddings,
            ]:
                embeddings.weight[token_idx] = rand_embedding
        self.ctx_bert.bert_model.embeddings.word_embeddings.weight.detach_()
        self.cand_bert.bert_model.embeddings.word_embeddings.weight.detach_()

    def forward(self, context_w, cands_w):
        if context_w is not None:
            context_segments = torch.zeros_like(context_w)
            context_mask = context_w != self.pad_idx
            context_h = self.ctx_bert(
                token_ids=context_w,
                segment_ids=context_segments,
                attention_mask=context_mask,
            )
            if self.opt.normalize_sent_emb:
                context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
        else:
            context_h = None
        if cands_w is not None:
            cands_segments = torch.zeros_like(cands_w)
            cands_mask = cands_w != self.pad_idx
            cands_h = self.cand_bert(
                token_ids=cands_w, segment_ids=cands_segments, attention_mask=cands_mask
            )
            if self.opt.normalize_sent_emb:
                cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)
        else:
            cands_h = None
        return context_h, cands_h
