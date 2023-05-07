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
"""PyTorch LXRT model."""

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, SmoothL1Loss, CosineEmbeddingLoss
from .pytorch_i3d import InceptionI3d
from .capsules_new_3d import PrimaryCaps, ConvCaps

# from capsules import PrimaryLinearCaps, LinearCaps, PrimaryLinearDynamicCaps, LinearDynamicCaps, get_activation, sep_caps
# from setTransformer import TransformerCapsules, PrimaryTransformerCapsules, TransformerEncoderLayer
# from ovr_cnn.maskrcnn_benchmark.modeling.mmss_heads.grounding_head import GroundingHead
# from ovr_cnn.maskrcnn_benchmark.config import cfg
from .file_utils import cached_path
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except Importtokenization:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualConfig(object):
    VISUAL_LOSSES = ['obj', 'attr', 'feat']
    def __init__(self,
                 l_layers=12,
                 x_layers=5,
                 r_layers=0,
                 num_prim_caps=32,
                 num_vis_caps=32,
                 pose_dim=4,
                 hw=7,
                 patches=False):
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        self.patches = patches #flag to train with image patches if true
        self.positional_encoding_type = "learned"
        self.visual_feat_dim = 2048
        if self.patches:
            self.visual_feat_dim = 3072
        self.visual_pos_dim = 4

        self.obj_id_num = 1600
        self.attr_id_num = 400

        self.hw = hw
        self.t = 8

        #capsules param
        self.num_prim_caps = num_prim_caps
        self.num_vis_caps = num_vis_caps
        self.pose_matrix_dim = pose_dim
        self.caps_dim = self.num_vis_caps * (self.pose_matrix_dim*self.pose_matrix_dim+1)

        self.max_spatial_pos_emb = self.t*hw*hw

        self.visual_losses = self.VISUAL_LOSSES
        self.visual_loss_config = {
            'obj': (self.obj_id_num, 'ce', (-1,), 1/0.15),
            'attr': (self.attr_id_num, 'ce', (-1,), 1/0.15),
            'feat': (2048, 'l2', (-1, 2048), 1/0.15),
        }

    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim


VISUAL_CONFIG = VisualConfig()


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 visualization=True):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.visualization = visualization
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.visualization = config.visualization

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            # print(attention_scores.shape, attention_mask.shape)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attn_data


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_probs = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_probs


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output, attention_probs = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""


class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)
        self.visualization = config.visualization


        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output, attention_probs_xl = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output, attention_probs_xv = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output, attention_probs_xl, attention_probs_xv

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output, attention_probs_l = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output, attention_probs_v = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output, attention_probs_l, attention_probs_v

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output, attn_prob_xl, attn_prob_xv = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output, attn_prob_l, attn_prob_v = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        attn_probs = {
            'attn_prob_l':attn_prob_l,
            'attn_prob_v':attn_prob_v,
            'attn_prob_xl':attn_prob_xl,
            'attn_prob_xv':attn_prob_xv

        }

        return lang_output, visn_output, attn_probs

class Linear(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super(Linear, self).__init__()
        self.dense = nn.Linear(in_features=in_dim, out_features=out_dim)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertReferExpHead(nn.Module):
    def __init__(self, hidden_size, out_dim1, out_dim2):
        super().__init__()
        hid_dim = hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Dropout(0.8),
        )
        self.box_regressor = nn.Linear(hid_dim * 2, out_dim1)
        self.box_regressor2 = nn.Linear(hid_dim * 2, out_dim2)

    def forward(self, hidden_states):
        x = self.logit_fc(hidden_states)
        xcycwh = self.box_regressor(x)
        box_params = self.box_regressor2(x)
        return xcycwh, box_params

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        # self.LayerNorm = BertLayerNorm(hid_dim * 2, eps=1e-12)
        self.dropout = nn.Dropout()
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #changed from F.relu to GeLU
            x = self.dropout(GeLU(layer(x))) if i < self.num_layers - 1 else layer(x)
        return x

######## cross attentional layers #######
class CrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # self.visn_self_att = BertSelfattLayer(config)
        self.visualization = config.visualization


    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output, attention_probs_xl = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output, attention_probs_xv = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output, attention_probs_xl, attention_probs_xv

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output


    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask, last=None):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output, attn_prob_xl, attn_prob_xv = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)

        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        attn_probs = {
            'attn_prob_l':[],
            'attn_prob_v':[],
            'attn_prob_xl':attn_prob_xl,
            'attn_prob_xv':attn_prob_xv,
            'attn_prob_vl': []

        }

        return lang_output, visn_output, attn_probs

class SelfCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.cross_att = BertSelfattLayer(config)

        # Intermediate and Output layer (FFNs)
        self.vl_inter = BertIntermediate(config)
        self.vl_output = BertOutput(config)

        self.visualization = config.visualization

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, step):
        #visn_input is actually visn+lang sequence
        #lang_input is either the original lang output, or just the hidden states for lang tokens
        # after self attention layer
        # Self Attention
        #todo: concat inputs and their attention masks and pass to self attn layer
        if step == 0:
            vis_lang_input = torch.cat([visn_input, lang_input], dim=1)
        else:
            # subsequent layers take output of previous layer which is one sequence, not two
            vis_lang_input = visn_input
        vis_lang_attention_mask = torch.cat([visn_attention_mask, lang_attention_mask], dim=-1)
        att_output, attention_probs = self.cross_att(vis_lang_input, vis_lang_attention_mask)
        # visn_att_output, attention_probs_v = self.visn_self_att(visn_input, visn_attention_mask)
        return att_output, attention_probs


    def output_fc(self, vl_input):
        # FC layers
        vl_inter_output = self.vl_inter(vl_input)

        # Layer output
        vl_output = self.vl_output(vl_inter_output, vl_input)
        return vl_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask, step=None):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        visn_input_len = visn_attention_mask.size(-1)

        vl_att_output, attn_prob_vl = self.self_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask, step=step)

        vl_output = self.output_fc(vl_att_output)

        attn_probs = {
            'attn_prob_l':[],
            'attn_prob_v':[],
            'attn_prob_xl':[],
            'attn_prob_xv':[],
            'attn_prob_vl': attn_prob_vl

        }
        #lang output is the hidden states for lang tokens after self attention on both lang and vision
        lang_output = vl_output[:,visn_input_len:]

        return lang_output, vl_output, attn_probs

class CrossAndSelfLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)
        # The cross-attention Layer
        self.self_att_layer = BertSelfattLayer(config)

        # Intermediate and Output layer (FFNs)
        self.vl_inter = BertIntermediate(config)
        self.vl_output = BertOutput(config)

        self.visualization = config.visualization

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output, attention_probs_xl = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output, attention_probs_xv = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output, attention_probs_xl, attention_probs_xv

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, step=None):
        # Self Attention
        # if step == 0:
        vis_lang_input = torch.cat([visn_input, lang_input], dim=1)
        # else:
        #     #subsequent layers take output of previous layer which is one sequence, not two
        #     vis_lang_input = visn_input

        vis_lang_attention_mask = torch.cat([visn_attention_mask, lang_attention_mask], dim=-1)
        att_output, attention_probs = self.self_att_layer(vis_lang_input, vis_lang_attention_mask)
        # visn_att_output, attention_probs_v = self.visn_self_att(visn_input, visn_attention_mask)
        return att_output, attention_probs


    def output_fc(self, vl_input):
        # FC layers
        vl_inter_output = self.vl_inter(vl_input)

        # Layer output
        vl_output = self.vl_output(vl_inter_output, vl_input)
        return vl_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask, step=None):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        visn_input_len = visn_attention_mask.size(-1)

        lang_att_output, visn_att_output, attn_prob_xl, attn_prob_xv = self.cross_att(lang_att_output,
                                                                                      lang_attention_mask,
                                                                                      visn_att_output,
                                                                                      visn_attention_mask)

        vl_att_output, attn_prob_vl = self.self_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask, step=step)

        vl_output = self.output_fc(vl_att_output)

        attn_probs = {
            'attn_prob_l':[],
            'attn_prob_v':[],
            'attn_prob_xl':[],
            'attn_prob_xv':[],
            'attn_prob_vl': attn_prob_vl

        }
        #lang output is the hidden states for lang tokens after self attention on both lang and vision
        lang_output = vl_output[:,visn_input_len:]
        visn_output = vl_output[:, :visn_input_len]

        return lang_output, visn_output, attn_probs




# class TrCapsuleNet5Act(nn.Module):  # Baseline Method
#     def __init__(
#             self,
#             embd_dim=1024,
#             video_dim=2048,
#             we_dim=300,
#             merge=True,
#             n_caps1=256,
#             p_dim1=16,
#             hidden_dim=4096,
#             recon=0
#     ):
#         super(TrCapsuleNet5Act, self).__init__()
#         self.DAVEnet = load_DAVEnet()
#
#         assert not merge
#
#         self.PrimCaps_audio = PrimaryTransformerCapsules(1024, n_caps1, p_dim1)
#
#         self.PrimCaps_video = PrimaryTransformerCapsules(video_dim, n_caps1, p_dim1)
#
#         self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim)
#         self.PrimCaps_text = PrimaryTransformerCapsules(embd_dim, n_caps1, p_dim1)
#
#         d_model = 16
#         self.capsule_expander = nn.Linear(p_dim1, d_model)
#
#         n_heads = 1
#         self.Transformer = TransformerEncoderLayer(d_model, n_heads, dim_feedforward=1024, dropout=0.1)
#         self.create_mask = nn.Linear(d_model, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(-2)
#
#         self.projection_v = nn.Linear(n_caps1 * p_dim1, hidden_dim)
#         self.projection_a = nn.Linear(n_caps1 * p_dim1, hidden_dim)
#         self.projection_t = nn.Linear(n_caps1 * p_dim1, hidden_dim)
#
#         self.recon = recon
#         if self.recon:
#             self.recon_v = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim // 8),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim // 8, video_dim),
#                 nn.ReLU(inplace=True)
#             )
#             self.recon_a = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim // 8),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim // 8, 1024),
#                 nn.ReLU(inplace=True)
#             )
#             self.recon_t = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim // 8),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim // 8, embd_dim),
#                 nn.ReLU(inplace=True)
#             )
#             self.mse = nn.MSELoss(reduction='none')
#
#         self.n_caps1 = n_caps1
#         self.p_dim1 = p_dim1
#
#         self.merge = merge
#     def save_checkpoint(self, path):
#         th.save(self.state_dict(), path)
#
#     def load_checkpoint(self, path):
#         try:
#             self.load_state_dict(th.load(path, map_location='cpu'))
#         except Exception as e:
#             print(e)
#             print("IGNORING ERROR, LOADING MODEL USING STRICT=FALSE")
#             self.load_state_dict(th.load(path, map_location='cpu'), strict=False)
#         print("Loaded model checkpoint from {}".format(path))
#
#     def forward(self, video, audio_input, nframes, text=None, sep_mod=False):
#         video_poses, video_acts = self.PrimCaps_video(video)
#
#         audio = self.DAVEnet(audio_input)
#         if not self.training:  # controlled by net.train() / net.eval() (use for downstream tasks)
#             # Mean-pool audio embeddings and disregard embeddings from input 0 padding
#             pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
#             nframes.div_(pooling_ratio)
#             audioPoolfunc = th.nn.AdaptiveAvgPool2d((1, 1))
#             audio_outputs = audio.unsqueeze(2)
#             pooled_audio_outputs_list = []
#             for idx in range(audio.shape[0]):
#                 nF = max(1, nframes[idx])
#                 pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
#             audio = th.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
#         else:
#             audio = audio.mean(dim=2)  # this averages features from 0 padding too
#
#         audio_poses, audio_acts = self.PrimCaps_audio(audio)
#
#         text = self.text_pooling_caption(text)
#         text_poses, text_acts = self.PrimCaps_text(text)
#
#         v_expanded = self.capsule_expander(video_poses)
#         a_expanded = self.capsule_expander(audio_poses)
#         t_expanded = self.capsule_expander(text_poses)
#
#         v_conditional = self.Transformer(v_expanded*video_acts.unsqueeze(-1))
#         a_conditional = self.Transformer(a_expanded*audio_acts.unsqueeze(-1))
#         t_conditional = self.Transformer(t_expanded*text_acts.unsqueeze(-1))
#
#         v_mask_logits = self.create_mask(v_conditional)
#         a_mask_logits = self.create_mask(a_conditional)
#         t_mask_logits = self.create_mask(t_conditional)
#         v_mask = self.softmax(v_mask_logits)
#         a_mask = self.softmax(a_mask_logits)
#         t_mask = self.softmax(t_mask_logits)
#
#         if self.training:
#             v_masked = video_poses * v_mask
#             a_masked = audio_poses * a_mask
#             t_masked = text_poses * t_mask
#         else:
#             v_masked = video_poses * v_mask
#             a_masked = audio_poses * a_mask
#             t_masked = text_poses * t_mask
#
#         v_masked = v_masked.view(-1, self.n_caps1*self.p_dim1)
#         video_out = self.projection_v(v_masked)
#
#         a_masked = a_masked.view(-1, self.n_caps1*self.p_dim1)
#         audio_out = self.projection_v(a_masked)
#
#         t_masked = t_masked.view(-1, self.n_caps1*self.p_dim1)
#         text_out = self.projection_v(t_masked)
#
#         return video_out, audio_out, text_out, v_mask[..., 0], a_mask[..., 0], t_mask[..., 0]
#

class VisualFeatEncoder(nn.Module):
    def __init__(self, config, norm_inputs=False, patches=False, no_caps=False):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim
        pos_dim = VISUAL_CONFIG.visual_pos_dim
        num_prim_caps = VISUAL_CONFIG.num_prim_caps
        num_vis_caps = VISUAL_CONFIG.num_vis_caps
        P = VISUAL_CONFIG.pose_matrix_dim

        self.patches = patches
        self.no_caps = no_caps
        img_dim = 224
        patch_dim = 32
        num_channels = 3
        assert img_dim % patch_dim == 0
        if self.patches:
            self.patch_dim = patch_dim
            self.num_patches = int((img_dim // patch_dim) ** 2)
            self.seq_length = self.num_patches + 1
            self.spatial_dim = int(math.sqrt(self.num_patches))
            self.flatten_dim = patch_dim * patch_dim * num_channels
            self.linear_encoding = nn.Linear(self.flatten_dim, config.hidden_size)
        else:
            #for clip_len T=16 (input feature shape (B, 2048, 16, 7, 7)), the 3d kernel (5,3,3) generates
            # output shape (B, hidden_size, 8, 7, 7) i.e. 392 visual tokens which is a manageable input to BERT
            self.conv3d = nn.Sequential(nn.ZeroPad2d(1),
                                        nn.Conv3d(2048, config.hidden_size, kernel_size=(5,3,3)),  #stride==kernel_size --> non-ovrlapping patches
                                        GeLU(),
                                        nn.ZeroPad2d(1),
                                        nn.Conv3d(config.hidden_size, config.hidden_size, kernel_size=(5,3,3)),  #stride==kernel_size --> non-ovrlapping patches
                                        GeLU())
        if self.no_caps:
            self.cls_token3d = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            # todo: load cls from vit model, if vit init is selected

            self.caps_dim = config.hidden_size
        else:

            self.caps_dim = num_vis_caps * (P * P + 1)
            self.is_attn_routing = False
            # Object feature encoding
            self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
            self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

            if self.is_attn_routing:
                #todo: initialize self attention capsules
                raise NotImplementedError

            else:
                self.primary_caps3d = PrimaryCaps(config.hidden_size, num_prim_caps, (1, 1, 1), P, stride=1)
                self.conv_caps = ConvCaps(num_prim_caps, num_vis_caps, (1, 1, 1), P, stride=(1, 1, 1), iters=3)
            self.cls_token3d = nn.Parameter(torch.zeros(1, 1, self.caps_dim))
        #todo: load cls from vit model, if vit init is selected
        self.seq_length = VISUAL_CONFIG.max_spatial_pos_emb + 1  # +1 for cls token
        # position embedding
        if VISUAL_CONFIG.positional_encoding_type == "learned":
            self.position_encoding3d = LearnedPositionalEncoding(
                self.seq_length, self.caps_dim, self.seq_length
            )
        elif VISUAL_CONFIG.positional_encoding_type == "fixed":
            self.position_encoding3d = FixedPositionalEncoding(
                self.caps_dim,
            )

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.pos_layer_norm3d = BertLayerNorm(self.caps_dim, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.normalize_inputs = norm_inputs

    def forward(self, visn_input):
        feats, boxes = visn_input #todo: change visn_input to return spatial features
        feats = feats.float()
        B = feats.shape[0]
        if self.patches:
            img = self.linear_encoding(feats)

            # print(img.shape)
            B, hw, dim = img.shape
            assert hw == self.num_patches
            img = img.view(B, dim, self.spatial_dim, self.spatial_dim)
            # img = img.view()
        else:
            # print(feats.shape)
            img = self.conv3d(feats)  # B, dim, H, W

        cls_tokens = self.cls_token3d.expand(B, -1, -1)
        if self.no_caps:
            _, D, H, W = img.shape
            x = torch.cat((cls_tokens, img.permute(0, 2, 3, 1).view(B, H * W, D)), dim=1)
        else:
            # print(img.shape)
            img_prim_caps = self.primary_caps3d(img)
            b, t, h, w, c = img_prim_caps.shape
            # print(img_prim_caps.shape)
            vis_caps, a_out = self.conv_caps(img_prim_caps.contiguous().view(-1, h, w, c)) # shape would be B, H, W, C, D

            BT, H, W, C, D = vis_caps.size()

            x = torch.cat((cls_tokens, vis_caps.view(b, t * H * W, C * D)), dim=1)

        x = self.position_encoding3d(x)

        output = self.dropout(x)
        return output, a_out


class NoCapsEncoder(nn.Module):
    def __init__(self, config, shared_weights=False, cross_attn=False, cross_attn_type= 'cross', no_caps=False, mask_features=False):
        super().__init__()
        self.no_caps = no_caps
        self.mask_features = mask_features
        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(config, no_caps=no_caps)
        self.cross_attn_layer = nn.ModuleDict({
            'cross': CrossLayer(config),
            'self': SelfCrossLayer(config),
            'cross_self': CrossAndSelfLayer(config),
            'old': CrossLayer(config)  # for compatibility with previously pretrained models
        })

        if self.mask_features:
            self.mask_capsules = nn.Linear(config.hidden_size, VISUAL_CONFIG.num_vis_caps)

        # Number of layers
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers
        print("LXRT encoder with %d l_layers, %d x_layers, and %d r_layers." %
              (self.num_l_layers, self.num_x_layers, self.num_r_layers))

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        self.x_layers = nn.ModuleList(
            [self.cross_attn_layer[cross_attn_type] for _ in range(self.num_x_layers)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        )

    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None,output_all_attention_masks=False):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats, _ = self.visn_fc(visn_feats)

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_xl = []
        all_attention_mask_xv = []
        all_attention_mask_vl = []

        #mask visn_feats
        if self.mask_features:
            mask_logits = self.mask_capsules(lang_feats[:,0])
            mask = torch.nn.functional.softmax(mask_logits, dim=-1)
            visn_feats = torch.cat(
                [visn_feats[:, 0, :, :].unsqueeze(1), visn_feats[:, 1:, :, :] * mask[:, None, :, None]], dim=1)

        # Run language layers
        for layer_module in self.layer:
            lang_feats, txt_attention_probs = layer_module(lang_feats, lang_attention_mask)

        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats, image_attention_probs = layer_module(visn_feats, visn_attention_mask)

        # Run cross-modality layers
        # if 'self' or 'cross_self' attention used, visn_feats are returned None,
        # and lang_feats are actually the output of self_attention layer on both visual and language inputs
        for layer_module in self.x_layers:
            lang_feats, visn_feats, attn_probs = layer_module(lang_feats, lang_attention_mask,
                                                  visn_feats, visn_attention_mask)
            if output_all_attention_masks:
                all_attention_mask_t.append(attn_probs['attn_prob_l'])
                all_attnetion_mask_v.append(attn_probs['attn_prob_v'])
                all_attention_mask_xl.append(attn_probs['attn_prob_xl'])
                all_attention_mask_xv.append(attn_probs['attn_prob_xv'])
                all_attention_mask_vl.append(attn_probs['attn_prob_vl'])

        return lang_feats, visn_feats, (all_attention_mask_t,
                                        all_attnetion_mask_v,
                                        all_attention_mask_xl,
                                        all_attention_mask_xv,
                                        all_attention_mask_vl)


class LXRTCapsulesEncoder(nn.Module):
    def __init__(self, config,
                 shared_weights=False,
                 cross_attn=False,
                 cross_attn_type='cross',
                 freeze_weights=False,
                 skip_connection=True,
                 patches=False,
                 vit_init=False,
                 start_index=0,
                 no_caps=False
                 ):
        super().__init__()
        self.shared_weights = shared_weights
        self.vit_init = vit_init


        #create cross attn layer
        self.cross_attn = cross_attn
        self.cross_attn_type = cross_attn_type
        self.cross_attn_layer = nn.ModuleDict({
            'cross': CrossLayer(config),
            'self': SelfCrossLayer(config),
            'cross_self': CrossAndSelfLayer(config),
            'old':CrossLayer(config) #for compatibility with previously pretrained models
        })

        self.skip_connection = skip_connection
        self.patches = patches
        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(config, patches=patches)
        print(VISUAL_CONFIG.num_prim_caps)

        self.vis2langFF = nn.Linear(VISUAL_CONFIG.caps_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.mask_capsules = nn.Linear(config.hidden_size, VISUAL_CONFIG.num_vis_caps)

        # Number of layers
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers
        print("LXRT encoder with %d l_layers, and %d r_layers." %
              (self.num_l_layers, self.num_r_layers))

        if self.cross_attn:
            print("LXRT encoder with %d x_layers." %
                  (self.num_x_layers))

        assert VISUAL_CONFIG.l_layers == VISUAL_CONFIG.r_layers


        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if self.cross_attn:
            self.x_layers = nn.ModuleList(
                [self.cross_attn_layer[cross_attn_type] for _ in range(self.num_x_layers)]
            )
        if not shared_weights:
            if self.vit_init:
                print("loading vit layers...")
                self.r_layers = self.load_vit_layers(self.num_r_layers, first_layer_index=start_index)
            else:
                self.r_layers = nn.ModuleList(
                    [BertLayer(config) for _ in range(self.num_r_layers)]
                )


    #this function loads pretrained vit model and returns num_layers based on start index
    # if first_layer_index=0, return first <num_layers> layers
    # to get last <num_layers> layers, set first_layer_index to 7 (i.e. 12-5 if num_layers=5)
    def load_vit_layers(self, num_layers, first_layer_index=0):

        vit = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=0)

        #first check, that we can start_index is valid to extract <num_layers> layers..
        # if first_layer_index is 8, you cannot load 5 layers starting from this index
        # as 8+5=13>12 (12 is total layers in Vit)
        assert self.num_r_layers + first_layer_index <= len(vit.blocks)
        r_layers = nn.ModuleList(
            vit.blocks[first_layer_index:first_layer_index+num_layers]
        )

        return r_layers

    def freeze_weights(self):
        for param in self.named_parameters():
            if 'x_layers' not in param[0]:
                param[1].requires_grad = False

    def get_masked_vis_feats(self, visn_layers, B, all_attention_mask_v, hw, k, lang_sent_enc, masked_vis_feats,
                             output_all_attention_masks, visn_attention_mask, visn_feats, skip_connection, masks):
        B, seq_len, c, d = visn_feats.shape
        for layer_module in visn_layers:
            if self.vit_init:
                masked_vis_feats, image_attention_probs = layer_module(masked_vis_feats)
            else:
                masked_vis_feats, image_attention_probs = layer_module(masked_vis_feats, visn_attention_mask)

            # skip connection
            if skip_connection:
                # get language mask from intermediate lang features
                mask_logits = self.mask_capsules(lang_sent_enc[k])
                k += 1
                mask = torch.nn.functional.softmax(mask_logits, dim=-1)

                masked_vis_intermediate = torch.cat(
                    [visn_feats[:, 0, :, :].unsqueeze(1), visn_feats[:, 1:, :, :] * mask[:, None, :, None]], dim=1)

                masked_vis_intermediate = self.vis2langFF(masked_vis_intermediate.view(B, seq_len, -1))

                # everytime we apply new lang mask on initial capsules
                # masked_vis_intermediate = visn_feats.view(B, hw, VISUAL_CONFIG.num_vis_caps, -1) * mask[:, None, :, None]
                #
                # masked_vis_intermediate = self.vis2langFF(masked_vis_intermediate.view(B, hw, -1))

                # skip connection from masked capsules to intermediate visn features
                masked_vis_feats = masked_vis_intermediate + masked_vis_feats

            if output_all_attention_masks:
                all_attention_mask_v.append(image_attention_probs)
                masks.append(mask)
        return masked_vis_feats, all_attention_mask_v, masks

    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None, output_all_attention_masks=False,):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats, a_out = self.visn_fc(visn_feats)
        # visn_feats_init = visn_feats.clone()
        all_attention_mask_t = []
        all_attention_mask_v = []
        all_attention_mask_xl = []
        all_attention_mask_xv = []
        all_attention_mask_vl = []
        activations = []
        masks = []

        #mask visn_feats
        mask_logits = self.mask_capsules(lang_feats[:,0])
        mask = torch.nn.functional.softmax(mask_logits, dim=-1)
        masks.append(mask)

        B, seq_len, cd = visn_feats.shape

        visn_feats = visn_feats.view(B, seq_len, VISUAL_CONFIG.num_vis_caps, -1)
        # not masking cls_token3d for visn_feats, hence, [:,1:,:,:]
        masked_vis_feats = torch.cat([visn_feats[:, 0, :, :].unsqueeze(1), visn_feats[:, 1:, :, :] * mask[:, None, :, None]], dim=1)

        masked_vis_feats = self.dropout(self.vis2langFF(masked_vis_feats.view(B, seq_len, -1)))

        lang_sent_enc = []

        # Run language layers
        for layer_module in self.layer:
            lang_feats, txt_attention_probs = layer_module(lang_feats, lang_attention_mask)
            lang_sent_enc.append(lang_feats[:,0]) #sent embedding is at position 0 [cls] token

            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)

        k = 0
        if self.shared_weights:
            # Run relational layers
            masked_vis_feats, all_attention_mask_v, masks = self.get_masked_vis_feats(self.layer, B, all_attention_mask_v, seq_len, k,
                                                                               lang_sent_enc, masked_vis_feats,
                                                                               output_all_attention_masks,
                                                                               visn_attention_mask, visn_feats,
                                                                               self.skip_connection, masks)
        else:
            # Run relational layers
            masked_vis_feats, all_attention_mask_v, masks = self.get_masked_vis_feats(self.r_layers, B, all_attention_mask_v,
                                                                               seq_len, k, lang_sent_enc, masked_vis_feats,
                                                         output_all_attention_masks, visn_attention_mask, visn_feats,
                                                                               self.skip_connection, masks)
        # Run cross-modality layers
        if self.cross_attn:
            for i, layer_module in enumerate(self.x_layers):
                lang_feats, masked_vis_feats, attn_probs = layer_module(lang_feats, lang_attention_mask,
                                                      masked_vis_feats, visn_attention_mask, i)
                if output_all_attention_masks:
                    all_attention_mask_xl.append(attn_probs['attn_prob_xl'])
                    all_attention_mask_xv.append(attn_probs['attn_prob_xv'])
                    all_attention_mask_vl.append(attn_probs['attn_prob_vl'])
                    activations.append(a_out)

        return lang_feats, masked_vis_feats, (all_attention_mask_t,
                                        all_attention_mask_v,
                                        all_attention_mask_xl,
                                        all_attention_mask_xv,
                                        all_attention_mask_vl,
                                        activations,
                                              masks)



class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, dummy_input=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. Second input is just to keep signatures same and is not used anywhere,.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPooler2(nn.Module):
    def __init__(self, config):
        super(BertPooler2, self).__init__()
        self.dense2 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states1, hidden_states2):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first tokens in both modalities
        first_token_tensor1 = hidden_states1[:, 0]
        first_token_tensor2 = hidden_states2[:, 0]
        first_token_tensors = torch.cat([first_token_tensor1, first_token_tensor2], dim=-1)
        pooled_output = self.dense2(first_token_tensors)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class BertVisualObjHead(nn.Module):
    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Decide the use of visual losses
        visual_losses = visual_losses.split(",")
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict({
            key: nn.Linear(config.hidden_size, VISUAL_CONFIG.visual_loss_config[key][0])
            for key in self.visual_losses
        })

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path == 'bert-base-uncased':
                try:
                    print("The BERT-weight-downloading query to AWS was time-out;" 
                          "trying to download from UNC servers")
                    archive_file = "https://nlp.cs.unc.edu/data/bert/bert-base-uncased.tar.gz"
                    resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
                except EnvironmentError:
                    print("The weight-downloading still crashed with link: %s, "
                          "please check your network connection" % archive_file)
                    return None
            else:
                logger.error(
                        "Model name '{}' was not found in model name list ({}). "
                        "We assumed '{}' was a path or url but couldn't find any file "
                        "associated to this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                            archive_file))
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        # if len(missing_keys) > 0:
        #     logger.info("Weights of {} not initialized from pretrained model: {}".format(
        #         model.__class__.__name__, missing_keys))
        # if len(unexpected_keys) > 0:
        #     logger.info("Weights from pretrained model not used in {}: {}".format(
        #         model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class NoCapsModel(BertPreTrainedModel):
    """LXRT Model."""

    def __init__(self, config, shared_weights=False, cross_attn=False, cross_attn_type='cross', no_caps=False):
        super().__init__(config)
        self.cross_attn_type = cross_attn_type
        self.embeddings = BertEmbeddings(config)
        self.encoder = NoCapsEncoder(config,
                                     shared_weights=shared_weights,
                                     cross_attn=cross_attn,
                                     cross_attn_type=cross_attn_type,
                                     no_caps=no_caps)

        self.pooler_dict = nn.ModuleDict({
            'cross': BertPooler2(config),
            'self': BertPooler(config),
            'cross_self': BertPooler(config),
            'no_cross': BertPooler2(config),
            'old': BertPooler(config)

        })

        self.pooler = self.pooler_dict[self.cross_attn_type]
        self.apply(self.init_bert_weights)



    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=input_ids.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        visual_attention_mask = visual_feats[1]

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=input_ids.dtype) # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run LXRT backbone
        lang_feats, visn_feats, attn_probs = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask)
        pooled_output = self.pooler(visn_feats, lang_feats)

        return (lang_feats, visn_feats), pooled_output, attn_probs

class LXRTCapsuleModel(BertPreTrainedModel):
    """LXRT Model."""

    def __init__(self, config,
                 shared_weights=False,
                 cross_attn=False,
                 cross_attn_type='cross',
                 skip_connection=True,
                 patches=False,
                 vit_init=False,
                 start_index=0,
                 no_caps=False,
                 ):
        super().__init__(config)
        self.cross_attn_type = cross_attn_type
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTCapsulesEncoder(config,
                                           shared_weights=shared_weights,
                                           cross_attn=cross_attn,
                                           cross_attn_type=cross_attn_type,
                                           skip_connection=skip_connection,
                                           patches=patches,
                                           vit_init=vit_init,
                                           start_index=start_index,
                                           no_caps=no_caps)

        self.pooler_dict = nn.ModuleDict({
            'cross': BertPooler2(config),
            'self': BertPooler(config),
            'cross_self': BertPooler(config),
            'no_cross': BertPooler2(config),
            'old':BertPooler(config)

        })

        self.pooler = self.pooler_dict[self.cross_attn_type]

        # if cross_attn_type == 'self':
        #     self.pooler = BertPooler(config)
        # elif cross_attn_type == 'cross':
        #     self.pooler = BertPooler2(config)
        # else:
        #     raise NotImplementedError()
        self.apply(self.init_bert_weights)

    def freeze_weights(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False

        self.encoder.freeze_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None, output_all_attention_masks=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=input_ids.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        visual_attention_mask = visual_feats[1]

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=input_ids.dtype) # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run LXRT backbone
        lang_feats, visn_feats, attn_probs = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask,
            output_all_attention_masks=output_all_attention_masks,
        )

        # if cross attention type is 'self' or 'cross_self', visn_feats are actually None (and not used),
        # pooler takes output of self attention from both modalities (coming from encoder),
        # and returns the first token hidden state. If cross_attn_type='cross', pooler processes first tokens
        # from both visual and language inputs
        if self.cross_attn_type == 'old':
            #to keep code compatible with the old model
            pooled_output = self.pooler(lang_feats, visn_feats)
        else:
            pooled_output = self.pooler(visn_feats, lang_feats)

        return (lang_feats, visn_feats), pooled_output, attn_probs


class LXRTPretraining(BertPreTrainedModel):
    def __init__(self,
                 config,
                 task_mask_lm=True,
                 task_matched=True,
                 task_obj_predict=True,
                 task_retrieval=False,
                 visual_losses='',
                 task_qa=True,
                 task_grounding = False,
                 task_contrastive=True,
                 num_answers=2,
                 skip_connection=False,
                 shared_weights=False,
                 cross_attn=False,
                 cross_attn_type='cross',
                 freeze_weights=False,
                 patches=False,
                 vit_init=False,
                 start_index=0,
                 no_caps=False,
                 margin=0.1):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_answers = num_answers

        # Use of pre-training tasks
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_matched = task_matched
        self.task_qa = task_qa
        self.task_retreival = task_retrieval
        self.task_grounding = task_grounding
        self.task_contrastive = task_contrastive
        self.freeze_weights = freeze_weights
        self.patches = patches
        self.no_caps = no_caps
        self.margin = margin #todo: add param in config
        self.vit_init = vit_init if hasattr(self,'vit_init') else False
        self.start_index = start_index if hasattr(self, 'start_index') else 0

        # LXRT backbone
        if no_caps:
            self.bert = NoCapsModel(config,
                                    cross_attn_type=cross_attn_type,
                                    no_caps=no_caps
                                    )
        else:
            self.bert = LXRTCapsuleModel(config,
                                         shared_weights=shared_weights,
                                         cross_attn=cross_attn,
                                         cross_attn_type = cross_attn_type,
                                         skip_connection=skip_connection,
                                         patches=patches,
                                         vit_init=vit_init,
                                         start_index=start_index,
                                         no_caps=no_caps)
        if self.freeze_weights:
            self.bert.freeze_weights()
        # else:
        #     self.bert = NoCapsModel(config,
        #                           shared_weights=shared_weights,
        #                           cross_attn=cross_attn)

        # Pre-training heads
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = BertVisualObjHead(config, visual_losses)
        if self.task_qa:
            self.answer_head = BertVisualAnswerHead(config, self.num_answers)
        if self.task_grounding:
            self.grounding_head = GroundingHead(config=cfg, v_dim=config.hidden_size, l_dim=config.hidden_size)

        # Weight initialization
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        (lang_output, visn_output), pooled_output, attn_probs = self.bert(
            input_ids, token_type_ids, attention_mask,
            visual_feats=(visual_feats, pos),
        )


        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            # This answer_score would not be used anywhere,
            # just to keep a constant return function signature.
            answer_score = pooled_output[0][0]

        total_loss = 0.
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss_fct_contrast = CosineEmbeddingLoss(margin=self.margin)

        losses = ()
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
            losses += (masked_lm_loss.detach(),)
        if matched_label is not None and self.task_matched:
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            losses += (matched_loss.detach(),)
        if ans is not None and self.task_qa:
            answer_loss = loss_fct(
                answer_score.view(-1, self.num_answers),
                ans.view(-1)
            )
            # Since this Github version pre-trains with QA loss from the beginning,
            # I exclude "*2" here to match the effect of QA losses.
            # Previous: (loss *0) for 6 epochs, (loss *2) for 6 epochs.   (Used 10 instead of 6 in EMNLP paper)
            # Now     : (loss *1) for 12 epochs
            #
            # * 2       # Multiply by 2 because > half of the data will not have label
            total_loss += answer_loss
            losses += (answer_loss.detach(),)

        if matched_label is not None and self.task_contrastive:
            contrastive_loss = loss_fct_contrast(
                lang_output[:,0],
                visn_output[:,0],
                target=matched_label
            )
            total_loss += contrastive_loss
            losses += (contrastive_loss.detach(),)
        if obj_labels is not None and self.task_obj_predict:
            loss_fcts = {
                'l2': SmoothL1Loss(reduction='none'),
                'ce': CrossEntropyLoss(ignore_index=-1, reduction='none')
            }
            total_visn_loss = 0.
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in VISUAL_CONFIG.visual_losses:
                label, mask_conf = obj_labels[key]
                output_dim, loss_fct_name, label_shape, weight = VISUAL_CONFIG.visual_loss_config[key]
                visn_loss_fct = loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(
                    visn_prediction_scores.view(-1, output_dim),
                    label.view(*label_shape),
                )
                if visn_loss.dim() > 1:     # Regression Losses
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
                losses += (visn_loss.detach(),)
            total_loss += total_visn_loss
        # print(losses, file=sys.stderr)
        return total_loss, torch.stack(losses).unsqueeze(0), answer_score.detach()


class LXRTFeatureExtraction(BertPreTrainedModel):
    """
    BERT model for classification.
    """
    def __init__(self, config,
                 mode='lxr',
                 skip_connection=False,
                 shared_weights=False,
                 cross_attn=False,
                 cross_attn_type='cross',
                 freeze_weights=False,
                 patches=False,
                 vit_init=False,
                 start_index=0,
                 no_caps=False,
                 margin=0.1):
        """

        :param config:
        :param mode:  Number of visual layers
        """
        super().__init__(config)
        self.freeze_weights = freeze_weights
        self.patches = patches
        # LXRT backbone
        #if skip_connection:
        if no_caps:
            self.bert = NoCapsModel(config,
                                    cross_attn_type=cross_attn_type,
                                    no_caps=no_caps
                                    )
        else:
            self.bert = LXRTCapsuleModel(config,
                                    shared_weights=shared_weights,
                                    cross_attn=cross_attn,
                                     cross_attn_type=cross_attn_type,
                                    skip_connection=skip_connection,
                                    patches=patches,
                                     vit_init=vit_init,
                                     start_index=start_index,
                                     no_caps=no_caps
                                     )
        if self.freeze_weights:
            self.bert.freeze_weights()
        # else:
        #     self.bert = NoCapsModel(config,
        #                           shared_weights=shared_weights,
        #                           cross_attn=cross_attn)
        self.mode = mode
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None,
                visual_attention_mask=None, output_all_attention_masks=False,):
        feat_seq, pooled_output, attn_probs = self.bert(input_ids, token_type_ids, attention_mask,
                                            visual_feats=visual_feats,
                                            visual_attention_mask=visual_attention_mask,
                                            output_all_attention_masks=output_all_attention_masks)
        if 'x' == self.mode:
            return pooled_output, attn_probs
        elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
            return feat_seq, pooled_output, attn_probs
        elif 'l' in self.mode or 'r' in self.mode:
            return feat_seq, attn_probs

