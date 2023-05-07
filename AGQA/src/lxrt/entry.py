# coding=utf-8
# Copyright 2019 project LXRT.
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
import numpy as np

import torch
import torch.nn as nn

from src.lxrt.tokenization import BertTokenizer
from src.lxrt.modeling_capsbert import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG, BertFeatureExtraction, DeafFeatureExtraction


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, targets=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.targets = targets


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


def convert_relations_to_features(rel_trplts_tokens, num_rel=8, num_situations=16, lengths=[], loss_hg_per_frame=False):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, rel_trplts) in enumerate(rel_trplts_tokens):
        flatten_tokens = np.array(rel_trplts.view(-1))
        # num_rel, num_situations = 8, 16
        segment_ids = np.array(torch.tensor([[j] * num_rel for j in range(num_situations)]).view(-1))
        if loss_hg_per_frame:
            unpad_targets = [rel_trplts[j, :l] for j, l in zip(list(range(num_situations)), lengths.tolist()[i])]
            # unpad_targets = [item.item() for sublist in unpad_targets for item in sublist]
        else:
            unpad_targets = [rel_trplts[j, :l] for j, l in zip(list(range(num_situations)), lengths.tolist()[i])]
            unpad_targets = [item.item() for sublist in unpad_targets for item in sublist]
        assert flatten_tokens.shape == segment_ids.shape

        features.append(
            InputFeatures(input_ids=flatten_tokens,
                          input_mask=None,
                          segment_ids=segment_ids,
                          targets=unpad_targets))
    return features

def convert_relations_to_features_test(rel_trplts_tokens, num_rel=8, num_situations=16, lengths=[], bsize=8):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for i in range(bsize):
        # num_rel, num_situations = 8, 16
        segment_ids = np.array(torch.tensor([[j] * num_rel for j in range(num_situations)]).view(-1))

        features.append(
            InputFeatures(input_ids=None,
                          input_mask=None,
                          segment_ids=segment_ids,
                          targets=None))
    return features

def generate_rel_target_mask(num_situations, num_rel):
    #this function returns a mask to decode all relations per situation at once by setting -inf to future situations
    # and 0 to current and previous situations
    mask = torch.triu(torch.full((num_situations, num_situations), float('-inf')), diagonal=1)
    full_mask = np.repeat(np.array(mask), num_rel, axis=1)
    tgt_mask = np.repeat(full_mask, num_rel, axis=0)

    return tgt_mask


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers

    #capsules config
    VISUAL_CONFIG.num_prim_caps = args.NUM_PRIM_CAPS
    VISUAL_CONFIG.num_vis_caps = args.NUM_VIS_CAPS
    VISUAL_CONFIG.pose_matrix_dim = args.POSE_DIM
    VISUAL_CONFIG.hw = args.HW
    VISUAL_CONFIG.caps_dim = args.NUM_VIS_CAPS * (args.POSE_DIM*args.POSE_DIM+1)
    VISUAL_CONFIG.is_attn_routing = args.attn_routing
    print(VISUAL_CONFIG.num_prim_caps)


class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)
        self.args = args
        self.mode = mode

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        cross_attn_type = args.cross_attn_type if hasattr(args, 'cross_attn_type') else 'old'

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased",
            mode=mode,
            skip_connection=args.skip_connection,
            shared_weights=args.shared_weights,
            cross_attn = args.cross_attn,
            cross_attn_type=cross_attn_type,
            freeze_weights = args.freeze_weights,
            patches=args.patches,
            margin=args.margin,
            vit_init=args.vit_init,
            start_index=args.start_index,
            no_caps = args.no_caps
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)



    def multi_gpu(self):
        self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        self.model.to("cuda")




    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None):
        input_ids, input_mask, segment_ids = sents[0], sents[1], sents[2]

        # print(feats[0].shape)
        if self.mode == 'lxr':
            feat, output, attn_probs = self.model(input_ids, segment_ids, input_mask,
                                            visual_feats=feats,
                                            visual_attention_mask=visual_attention_mask,
                                            output_all_attention_masks=self.args.output_attention)
        else:
            feat = None
            output, attn_probs = self.model(input_ids, segment_ids, input_mask,
                                visual_feats=feats,
                                visual_attention_mask=visual_attention_mask, output_all_attention_masks=self.args.output_attention)
        return feat, output, attn_probs

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path, map_location=torch.device(self.device))
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value

            else:
                new_state_dict[key] = value
            if key.startswith("lxrt_encoder.model."):
                new_state_dict[key[len("lxrt_encoder.model."):]] = value
            # else:
            #     new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)








"""*** Q-ONLY MODEL ***"""
class BertTextEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)
        self.args = args
        self.mode = mode

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        cross_attn_type = args.cross_attn_type if hasattr(args, 'cross_attn_type') else 'old'

        # Build LXRT Model
        self.model = BertFeatureExtraction.from_pretrained(
            "bert-base-uncased",
            mode=mode,
            skip_connection=args.skip_connection,
            shared_weights=args.shared_weights,
            freeze_weights=args.freeze_weights,
            patches=args.patches,
            margin=args.margin,
            vit_init=args.vit_init,
            start_index=args.start_index,
            no_caps=args.no_caps
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

        # GPU Options
        # if torch.cuda.is_available():
        #     self.model = self.model.to(device)
        #     if args.multiGPU:
        #         self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        #         self.model.to(device)


    # def multi_gpu(self):
    #     self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents):
        input_ids, input_mask, segment_ids = sents[0], sents[1], sents[2]

        feat_seq, output, attn_probs = self.model(input_ids, segment_ids, input_mask,
                                                  output_all_attention_masks=self.args.output_attention)

        # output = [CLS] token only
        return feat_seq, output, attn_probs




    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path, map_location=torch.device(self.device))
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value

            else:
                new_state_dict[key] = value
            if key.startswith("lxrt_encoder.model."):
                new_state_dict[key[len("lxrt_encoder.model."):]] = value
            # else:
            #     new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)






""" *** DEAF MODEL ***"""
class DeafEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)
        self.args = args
        self.mode = mode


        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        cross_attn_type = args.cross_attn_type if hasattr(args, 'cross_attn_type') else 'old'

        # Build LXRT Model
        self.model = DeafFeatureExtraction.from_pretrained(
            "bert-base-uncased",
            mode=mode,
            skip_connection=args.skip_connection,
            shared_weights=args.shared_weights,
            cross_attn = args.cross_attn,
            cross_attn_type=cross_attn_type,
            freeze_weights = args.freeze_weights,
            patches=args.patches,
            margin=args.margin,
            vit_init=args.vit_init,
            start_index=args.start_index,
            no_caps = args.no_caps
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)




    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None):
        input_ids, input_mask, segment_ids = sents[0], sents[1], sents[2]


        # print(feats[0].shape)
        if self.mode == 'lxr':
            feat, output, attn_probs = self.model(input_ids, segment_ids, input_mask,
                                            visual_feats=feats,
                                            visual_attention_mask=visual_attention_mask,
                                            output_all_attention_masks=self.args.output_attention)
        else:
            feat = None
            output, attn_probs = self.model(input_ids, segment_ids, input_mask,
                                visual_feats=feats,
                                visual_attention_mask=visual_attention_mask, output_all_attention_masks=self.args.output_attention)
        return feat, output, attn_probs

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path, map_location=torch.device(self.device))
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value

            else:
                new_state_dict[key] = value
            if key.startswith("lxrt_encoder.model."):
                new_state_dict[key[len("lxrt_encoder.model."):]] = value
            # else:
            #     new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)