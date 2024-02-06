# coding=utf-8
# Copyleft 2019 project LXRT.

import torch
import torch.nn as nn

from src.param import args
from src.lxrt.entry import LXRTEncoder, generate_rel_target_mask
from src.lxrt.transformer import TransformerDecoder, TransformerDecoderLayer
from src.lxrt.modeling_capsbert import BertLayerNorm, GeLU, HGEmbeddings, CrossEncoder, BERT
from src.video_encoder import VideoBackbone


# Max length including <bos> and <eos>
MAX_STAR_LENGTH = 40


class STARModel(nn.Module):
    def __init__(self, num_answers, num_queries=128, num_classes=563, num_actions=111, model_name='', act_queries=48):
        super().__init__()

        self.vid_encoder = VideoBackbone(args.backbone) #possible backbones are 'slow_r50', 'resnext101', 'video_swin'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # # Set to eval mode and move to desired device
        self.vid_encoder.to(device)


        self.max_seq_length = MAX_STAR_LENGTH
        self.num_queries = num_queries if not args.gt_hg else num_classes+1
        self.act_queries = act_queries if not args.gt_hg else num_actions+1

        mode = 'l' if args.backbone == 'mvit_B' else 'lxr'
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_STAR_LENGTH,
            mode=mode
        )
        # assert (args.task_hgvqa or args.task_hgqa) and (args.add_action or args.add_relation), "HG should have either actions or relations or both."
        self.hgq_encoder = CrossEncoder(config=self.lxrt_encoder.model.config, cross_attn_type=args.cross_attn_type,
                                        num_max_act=args.num_act, num_max_rel=args.num_rel)
        hid_dim = self.lxrt_encoder.dim
        self.hid_dim = hid_dim
        self.relation_query_embed = HGEmbeddings(num_queries=self.num_queries, type_vocab_size=16, hidden_size=hid_dim, gt_hg=args.gt_hg)

        #16*3 --> num_situations * max_num_actions
        self.action_query_embed = HGEmbeddings(num_queries=self.act_queries, type_vocab_size=16, hidden_size=hid_dim,
                                               hidden_dropout_prob=args.emb_drop_rate, gt_hg=args.gt_hg)

        decoder_layer = TransformerDecoderLayer(d_model=hid_dim, nhead=12, dropout=args.decoder_drop_rate)
        #relation decoder
        self.rel_decoder = TransformerDecoder(decoder_layer, num_layers=args.dlayers)

        #relation classifier
        if args.linear_cls:
            self.class_embed = nn.Linear(hid_dim, num_classes + 1)
        else:
            self.class_embed = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_classes+1)
            )

        # action decoder
        self.action_decoder = TransformerDecoder(decoder_layer, num_layers=args.dlayers)

        # action classifier
        if args.linear_cls:
            self.action_embed = nn.Linear(hid_dim, num_actions + 1)
        else:
            self.action_embed = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_actions + 1)
            )

        if args.task_hgvqa:
            self.logit_fc2 = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_answers)
            )

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )

        self.logit_fc.apply(self.init_bert_weights)
        self.rel_decoder.apply(self.init_bert_weights)
        self.action_decoder.apply(self.init_bert_weights)

        self.args = args

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02) #hard coded value to move function here
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def forward(self, feat, pos, input_ids, input_masks, segment_ids, rel_segment_ids, rel_tgt_mask, act_segment_ids,
                act_tgt_mask, hg_mask, rel_tgt_ids=None, act_tgt_ids=None):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        B, T = feat.shape[0], feat.shape[2]

        feat = self.vid_encoder.encode(feat)
        if self.args.backbone == 'mvit_B':

            memory = feat
            feats, x, attn_probs = self.lxrt_encoder((input_ids, input_masks, segment_ids), (feat, pos))
            lang_feats, lang_attn_mask = attn_probs[-1]
            logit = self.logit_fc(x[0][:, 0])
        else:
            B, _, _, H, W = feat.shape

            feats, x, attn_probs = self.lxrt_encoder((input_ids, input_masks, segment_ids), (feat, pos))
            logit = self.logit_fc(x)

            if args.after_cross_attn_feats:
                lang_feats = feats[0]
                memory = feats[1] # visual features from cross attention module
            else:
                lang_feats, lang_attn_mask, memory, _ = attn_probs[-1]

        rel_tgt_mask = torch.tensor(generate_rel_target_mask(num_situations=args.num_situations, num_rel=args.num_rel)).to(device=rel_segment_ids.device)

        if args.gt_hg and (rel_tgt_ids is not None and act_tgt_ids is not None):
            query_embed = self.relation_query_embed(rel_segment_ids, rel_tgt_ids)
            action_embed = self.action_query_embed(act_segment_ids, act_tgt_ids)
            # prepare hg to input to lxrt_encoder
            hg_in = torch.cat([action_embed.squeeze(0).view(B, T, -1, self.hid_dim),
                               query_embed.squeeze(0).view(B, T, -1, self.hid_dim)],
                              dim=2
                              )

            rel_preds, act_preds = None, None

        else:
            query_embed = self.relation_query_embed(rel_segment_ids)
            tgt = torch.zeros_like(query_embed).to(device=query_embed.device)
            out = self.rel_decoder(tgt.permute(1,0,2), memory.permute(1,0,2), query_pos=query_embed.permute(1,0,2), tgt_mask=rel_tgt_mask)
            out = out.transpose(1, 2)
            rel_preds = self.class_embed(out)

            #decode actions from video features
            act_tgt_mask = torch.tensor(
                 generate_rel_target_mask(num_situations=args.num_situations, num_rel=args.num_act)).to(device=act_segment_ids.device)
            # todo: input act_tgt_ids and act_segment_ids to action_query_embed
            action_embed = self.action_query_embed(act_segment_ids)
            act_tgt = torch.zeros_like(action_embed).to(device=action_embed.device)
            act_out = self.action_decoder(act_tgt.permute(1, 0, 2), memory.permute(1, 0, 2), query_pos=action_embed.permute(1, 0, 2),
                                   tgt_mask=act_tgt_mask)
            act_out = act_out.transpose(1, 2)
            act_preds = self.action_embed(act_out)


            #prepare hg to input to lxrt_encoder
            hg_in = torch.cat([act_out.squeeze(0).view(B, T, -1, self.hid_dim),
                               out.squeeze(0).view(B, T, -1, self.hid_dim)],
                              dim=2
                              )

            rel_preds, act_preds = rel_preds.squeeze(0), act_preds.squeeze(0)
        if args.use_hg_mask:
            hg_mask = hg_mask.view(B, -1)
        else:
            hg_mask = None

        if args.task_hgvqa:
            x_hg, attn_probs = self.hgq_encoder(lang_feats, lang_attn_mask, hg_in.view(B, -1, self.hid_dim), hg_mask)
            hg_logit = self.logit_fc2(torch.cat([x, x_hg], dim=-1))
        else:
            x_hg, attn_probs = self.hgq_encoder(lang_feats, lang_attn_mask, hg_in.view(B, -1, self.hid_dim), hg_mask)
            hg_logit = self.logit_fc(x_hg)


        return logit, rel_preds, act_preds, hg_logit, attn_probs


