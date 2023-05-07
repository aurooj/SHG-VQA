# coding=utf-8
# Copyleft 2019 project LXRT.
import torch
import torch.nn as nn

from src.param import args
from src.lxrt.entry import LXRTEncoder, generate_rel_target_mask, BertTextEncoder, DeafEncoder
from src.lxrt.transformer import TransformerDecoder, TransformerDecoderLayer
from src.lxrt.modeling_capsbert import BertLayerNorm, GeLU, HGEmbeddings, CrossEncoder
from src.video_encoder import VideoBackbone


# Max length including <bos> and <eos>
MAX_STAR_LENGTH = 40


class AGQAModel(nn.Module):
    def __init__(self, num_answers, num_queries=128, num_classes=456, num_actions=156, model_name=''):
        super().__init__()
        #load res3Dnet
        # self.vid_encoder = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        # self.vid_encoder.blocks[-1] = nn.Identity() #replacing classification head with identity() block
        # # Set to eval mode and move to desired device
        # self.vid_encoder = self.vid_encoder.eval()

        self.vid_encoder = VideoBackbone(args.backbone) #possible backbones are 'slow_r50', 'resnext101', 'video_swin'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # # Set to eval mode and move to desired device
        self.vid_encoder.to(device)


        self.max_seq_length = MAX_STAR_LENGTH
        self.num_queries = num_queries if not args.gt_hg else num_classes+1
        self.act_queries = 16*3 if not args.gt_hg else num_actions+1


        if args.task_q:
            self.bert_encoder = BertTextEncoder(args,
                                                max_seq_length=MAX_STAR_LENGTH,
                                                mode='lxr'
                                                )
            hid_dim = self.bert_encoder.dim
            self.hid_dim = hid_dim


        elif args.task_vqa:
            self.lxrt_encoder = LXRTEncoder(
                args,
                max_seq_length=MAX_STAR_LENGTH,
            )

            hid_dim = self.lxrt_encoder.dim
            self.hid_dim = hid_dim


        else:
            if args.task_hgqa:
                self.lxrt_encoder = LXRTEncoder(
                    args,
                    max_seq_length=MAX_STAR_LENGTH,
                    mode='lxr'
                )
    
    
                config = self.lxrt_encoder.model.config
                self.hgq_encoder = CrossEncoder(config=config, cross_attn_type=args.cross_attn_type,
                                                num_max_act=args.num_act, num_max_rel=args.num_rel)
                hid_dim = self.lxrt_encoder.dim
                self.hid_dim = hid_dim
                


            elif args.task_vhga:
                self.deaf_encoder = DeafEncoder(
                    args,
                    max_seq_length=MAX_STAR_LENGTH, mode='lxr'
                )

                config = self.deaf_encoder.model.config
                self.hgq_encoder = CrossEncoder(config=config, cross_attn_type=args.cross_attn_type,
                                                num_max_act=args.num_act, num_max_rel=args.num_rel)

                hid_dim = self.deaf_encoder.dim
                self.hid_dim = hid_dim





            self.relation_query_embed = HGEmbeddings(num_queries=self.num_queries, type_vocab_size=16, hidden_size=hid_dim, gt_hg=args.gt_hg)

            #16*3 --> num_situations * max_num_actions
            self.action_query_embed = HGEmbeddings(num_queries=self.act_queries, type_vocab_size=16, hidden_size=hid_dim,
                                                   hidden_dropout_prob=args.emb_drop_rate, gt_hg=args.gt_hg)


            #relation decoder
            decoder_layer = TransformerDecoderLayer(d_model=hid_dim, nhead=12, dropout=args.decoder_drop_rate)
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

            self.rel_decoder.apply(self.init_bert_weights)
            self.action_decoder.apply(self.init_bert_weights)




        # answer classifier
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )


        self.logit_fc.apply(self.init_bert_weights)
        # self.rel_decoder.apply(self.init_bert_weights)
        # self.action_decoder.apply(self.init_bert_weights)

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

        if args.task_q:
            feats, x, attn_probs = self.bert_encoder((input_ids, input_masks, segment_ids))
            logit = self.logit_fc(x)
            return logit, attn_probs


        if args.task_vqa:
            feat = self.vid_encoder.encode(feat)
            B, _, T, H, W = feat.shape


            feats, x, attn_probs = self.lxrt_encoder((input_ids, input_masks, segment_ids), (feat, pos))
            logit = self.logit_fc(x)
            return logit, attn_probs



        else:
            feat = self.vid_encoder.encode(feat)
            B, _, T, H, W = feat.shape


            if args.task_hgqa:
                # device = 'cuda:' + str(feat.get_device()) if torch.cuda.is_available() else 'cpu'
                # pos = torch.ones(T * H * W, device=device)
                feats, x, attn_probs = self.lxrt_encoder((input_ids, input_masks, segment_ids), (feat, pos))
                logit = self.logit_fc(x)


            elif args.task_vhga:
                feats, x, attn_probs = self.deaf_encoder((input_ids, input_masks, segment_ids), (feat, pos))
                logit = self.logit_fc(x)



            if args.after_cross_attn_feats:
                lang_feats = feats[0]
                memory = feats[1] # visual features from cross attention module
            else:
                lang_feats, lang_attn_mask, memory, _ = attn_probs[-1]

            rel_tgt_mask = torch.as_tensor(generate_rel_target_mask(num_situations=args.num_situations, num_rel=args.num_rel)).to(device=rel_segment_ids.device)
            #todo: input rel_tgt_ids and rel_segment_ids to relation_query_embed
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
                act_tgt_mask = torch.as_tensor(
                     generate_rel_target_mask(num_situations=args.num_situations, num_rel=args.num_act)).to(device=act_segment_ids.device)
                # todo: input act_tgt_ids and act_segment_ids to action_query_embed
                action_embed = self.action_query_embed(act_segment_ids)
                act_tgt = torch.zeros_like(action_embed).to(device=action_embed.device)
                act_out = self.action_decoder(act_tgt.permute(1, 0, 2), memory.permute(1, 0, 2), query_pos=action_embed.permute(1, 0, 2),
                                       tgt_mask=act_tgt_mask)
                act_out = act_out.transpose(1, 2)
                act_preds = self.action_embed(act_out)



                #todo: debug following code
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
            x, attn_probs = self.hgq_encoder(lang_feats, lang_attn_mask, hg_in.view(B, -1, self.hid_dim), hg_mask)
            hg_logit = self.logit_fc(x)


            return logit, rel_preds, act_preds, hg_logit, attn_probs


