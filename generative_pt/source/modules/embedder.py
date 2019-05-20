#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/encoders/embedder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Embedding):
    """
    Embedder
    """
    def forward(self, input, unk_idx):
        idx = torch.gt(input, self.num_embeddings-1).long()
        # print(input[idx])
        _idx = idx.eq(0).long()
        input = torch.mul(input, _idx) + idx * unk_idx
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def load_embeddings(self, embeds, scale=0.05):
        """
        load_embeddings
        """
        assert len(embeds) == self.num_embeddings

        embeds = torch.tensor(embeds)
        num_known = 0
        for i in range(len(embeds)):
            if len(embeds[i].nonzero()) == 0:
                nn.init.uniform_(embeds[i], -scale, scale)
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        print("{} words have pretrained embeddings".format(num_known),
              "(coverage: {:.3f})".format(num_known / self.num_embeddings))
