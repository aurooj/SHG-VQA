# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import numpy as np
import gc
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from src.param import args
from src.lxrt.tokenization import BertTokenizer
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.star_model import STARModel
from src.visualization_tools.vis_utils import accuracy
from src.lxrt.entry import convert_sents_to_features, convert_relations_to_features, generate_rel_target_mask, convert_relations_to_features_test
from src.lxrt.matcher import HungarianMatcher

if args.patches:
    raise NotImplementedError
else:
    from src.tasks.star_data import STARDataset, STARTorchDataset, STAREvaluator

print(args)
#dictionary to hold indices to extract attention from based on the cross attn type
attn_idx = {
    'cross': 2,
    'cross_self':4,
    'old': 2,
    'self': 4
}
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = STARDataset(splits)
    tset = STARTorchDataset(dset)
    evaluator = STAREvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        # collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class STAR:
    def __init__(self):
        self.task_vqa = args.task_vqa
        self.task_hgqa = args.task_hgqa
        self.task_hgvqa = args.task_hgvqa
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = args.batch_size #2048 if args.multiGPU else args.batch_size#512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=True
            )
        else:
            self.valid_tuple = None

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        self.background_idx = 0
        self.clip_len = args.CLIP_LEN
        self.num_situations = self.train_tuple.dataset.num_situations
        self.num_rel = self.train_tuple.dataset.num_rel
        self.num_actions = len(self.train_tuple.dataset.action_classes)
        self.num_act = self.train_tuple.dataset.num_act

        self.model = STARModel(self.train_tuple.dataset.num_answers, num_queries=self.num_rel * self.num_situations,
                               num_actions=len(self.train_tuple.dataset.action_classes), act_queries=self.num_act*self.num_situations)

        for p in self.model.vid_encoder.parameters():
            p.requires_grad = False
        self.max_seq_length = self.model.max_seq_length
        self.rel_classes = 563
        self.eos_coef_rel = 0.1
        empty_weight = torch.ones(self.rel_classes + 1)
        empty_weight[self.background_idx] = self.eos_coef_rel #ideally should be set for the empty class index (0 in our case)
        self.empty_weight = empty_weight

        self.eos_coef_acts = 0.1
        empty_weight_acts = torch.ones(self.num_actions + 1)
        empty_weight_acts[self.background_idx] = self.eos_coef_acts
        self.empty_weight_acts = empty_weight_acts
        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)


        # GPU options
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if args.multiGPU:
                self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        #relation loss
        self.matcher = HungarianMatcher(cost_class=1, loss_hg_per_frame=args.loss_hg_per_frame, clip_len=self.clip_len)
        self.weight_dict = {"loss_ce": 1, "loss_bbox": 0}
        self.weight_dict["loss_giou"] = 0

        losses = ["labels"]

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from src.lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output

        os.makedirs(self.output, exist_ok=True)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_target_classes(self, outputs, targets, indices, empty_weight, log=True,  loss_hg_per_frame=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        acc = 0.
        idx = self._get_src_permutation_idx(indices)
        if loss_hg_per_frame:
            flat_tgts = [item for sublist in targets for item in sublist["labels"]]
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(flat_tgts, indices)])
            b, num_queries = src_logits.shape[:2]
            assert num_queries%self.clip_len == 0
            src_logits = src_logits.reshape(b*self.clip_len, num_queries//self.clip_len, -1)
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.background_idx,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if log:
            acc = accuracy(src_logits[idx], target_classes_o)[0]
        return target_classes, acc

    def loss_labels(self, outputs, targets, indices, empty_weight, log=True, loss_hg_per_frame=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        if loss_hg_per_frame:
            flat_tgts = [item for sublist in targets for item in sublist["labels"]]
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(flat_tgts, indices)])
            b, num_queries = src_logits.shape[:2]
            assert num_queries%self.clip_len == 0
            src_logits = src_logits.reshape(b*self.clip_len, num_queries//self.clip_len, -1)
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.background_idx,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight.to(device=src_logits.device))
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            # log_str = ''
            quesid2ans = {}

            for i, (ques_id, vid_id, feats, boxes, sent, rel_triplets, lengths, act_tokens, hg_mask,
                    qa0, qa1, qa2, qa3, act_lengths, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                if args.multiGPU:
                    self.model.module.vid_encoder.eval()
                else:
                    self.model.vid_encoder.eval()
                self.optim.zero_grad(set_to_none=True)


                train_features = convert_sents_to_features(
                    sent, self.max_seq_length, self.tokenizer)

                #process relation triplets
                rel_features = convert_relations_to_features(rel_triplets, num_rel=self.num_rel,
                                                             num_situations=self.num_situations, lengths=lengths,
                                                             loss_hg_per_frame=args.loss_hg_per_frame)
                rel_feat_tgt_mask = torch.tensor(generate_rel_target_mask(num_situations=self.num_situations, num_rel=self.num_rel))

                rel_input_ids = torch.tensor(np.array([f.input_ids for f in rel_features]), dtype=torch.long)
                rel_segment_ids = torch.tensor(np.array([f.segment_ids for f in rel_features]), dtype=torch.long)
                if args.loss_hg_per_frame:
                    tgts = [{"labels": f.targets} for f in rel_features]
                else:
                    tgts = [{"labels": torch.tensor(f.targets)} for f in rel_features]

                #process action labels
                act_features = convert_relations_to_features(act_tokens, num_rel=self.num_act,
                                                             num_situations=self.num_situations, lengths=act_lengths,
                                                             loss_hg_per_frame=args.loss_hg_per_frame)
                act_feat_tgt_mask = torch.tensor(
                    generate_rel_target_mask(num_situations=self.num_situations, num_rel=self.num_act))

                act_input_ids = torch.tensor(np.array([f.input_ids for f in act_features]), dtype=torch.long)
                act_segment_ids = torch.tensor(np.array([f.segment_ids for f in act_features]), dtype=torch.long)
                if args.loss_hg_per_frame:
                    act_tgts = [{"labels": f.targets} for f in act_features]
                else:
                    act_tgts = [{"labels": torch.tensor(f.targets)} for f in act_features]

                #process question features
                input_ids = torch.tensor(np.array([f.input_ids for f in train_features]), dtype=torch.long)
                input_mask = torch.tensor(np.array([f.input_mask for f in train_features]), dtype=torch.long)
                segment_ids = torch.tensor(np.array([f.segment_ids for f in train_features]), dtype=torch.long)

                if torch.cuda.is_available():
                    feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                    input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                    rel_input_ids, rel_segment_ids, rel_feat_tgt_mask = rel_input_ids.cuda(), rel_segment_ids.cuda(), rel_feat_tgt_mask.cuda()
                    hg_mask = hg_mask.cuda()
                    if args.loss_hg_per_frame:
                        tgts = [{"labels": [g.cuda() for g in f.targets]} for f in rel_features]
                    else:
                        tgts = [{"labels": torch.tensor(f.targets).cuda()} for f in rel_features]

                    act_input_ids, act_segment_ids, act_feat_tgt_mask = act_input_ids.cuda(), act_segment_ids.cuda(), act_feat_tgt_mask.cuda()
                    if args.loss_hg_per_frame:
                        act_tgts = [{"labels": [g.cuda() for g in f.targets]} for f in act_features]
                    else:
                        act_tgts = [{"labels": torch.tensor(f.targets).cuda()} for f in act_features]

                if args.gt_hg:
                    rel_tgt_ids = rel_input_ids
                    act_tgt_ids = act_input_ids
                else:
                    rel_tgt_ids, act_tgt_ids = None, None

                logit, rel_logit, act_logit, hg_logit, attn_probs = self.model(feats, boxes, input_ids=input_ids, input_masks=input_mask,
                                                          segment_ids=segment_ids, rel_segment_ids=rel_segment_ids,
                                                          rel_tgt_mask=rel_feat_tgt_mask, act_segment_ids=act_segment_ids,
                                                          act_tgt_mask=act_feat_tgt_mask, hg_mask=hg_mask,
                                                          rel_tgt_ids=rel_tgt_ids, act_tgt_ids=act_tgt_ids)



                assert logit.dim() == target.dim() == 2
                losses = ()
                total_loss = 0.
                vqa_loss_str = ""
                hgqa_loss_str = ""
                hg_loss_str = ""
                if self.task_vqa:
                    if args.mce_loss:
                        max_value, target = target.max(1)
                        loss = self.mce_loss(logit, target) * logit.size(1)
                    else:
                        loss = self.bce_loss(logit, target)
                        loss = loss * logit.size(1)
                    total_loss += loss
                    losses += (loss.detach(),)
                    vqa_loss_str = "\tVQA loss= %0.4f"%loss.detach().item()

                if self.task_hgqa or self.task_hgvqa:
                    task = "HGQA" if self.task_hgqa else "HGVQA"
                    hgqa_loss = self.bce_loss(hg_logit, target)
                    hgqa_loss = hgqa_loss * hg_logit.size(1)
                    total_loss +=hgqa_loss
                    losses += (hgqa_loss.detach(),)
                    hgqa_loss_str = "\t%s loss= %0.4f"% (task, hgqa_loss.detach().item())

                    if not args.gt_hg:
                        # compute hungarian matching loss between predicted relation and target relations
                        # Retrieve the matching between the outputs of the last layer and the targets
                        indices = self.matcher({'pred_logits': rel_logit}, tgts,
                                               )
                        rel_loss = self.loss_labels({'pred_logits': rel_logit}, tgts, indices,
                                                    empty_weight=self.empty_weight, loss_hg_per_frame=args.loss_hg_per_frame)

                        # compute hungarian matching loss between predicted actions and target actions
                        # Retrieve the matching between the outputs of the last layer and the targets
                        act_indices = self.matcher({'pred_logits': act_logit}, act_tgts)
                        act_loss = self.loss_labels({'pred_logits': act_logit}, act_tgts, act_indices,
                                                    empty_weight=self.empty_weight_acts, loss_hg_per_frame=args.loss_hg_per_frame)

                        #add relation loss
                        for los, loss_value in rel_loss.items():
                            if los in self.weight_dict:
                                los_w = self.weight_dict[los] * loss_value
                                total_loss += los_w
                                losses += (los_w.detach(),)

                        #add action loss
                        for los, loss_value in act_loss.items():
                            if los in self.weight_dict:
                                los_w = self.weight_dict[los] * loss_value
                                total_loss += los_w
                                losses += (los_w.detach(),)

                        hg_loss_str = "\tRel loss= %0.4f \tAct loss= %0.4f\n" \
                                      "Rel class error= %0.4f \t Act class error= %0.4f\n" % (
                                      rel_loss["loss_ce"].item(), act_loss["loss_ce"].item(),
                                      rel_loss["class_error"].item(), act_loss["class_error"].item())

                if i % args.log_freq == 0:
                    log_loss = "\nEpoch %d: Total loss= %0.4f "% (epoch, total_loss.item())

                    log_loss += vqa_loss_str + hgqa_loss_str + hg_loss_str

                    print(log_loss)

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = hg_logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    # ans = dset.label2ans[l]
                    ans = l
                    quesid2ans[qid] = ans


            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)


            if self.valid_tuple is not None:  # Do Validation
                valid_score, hg_score = self.evaluate(eval_tuple) #todo: keep best model based on hg_score
                if (self.task_hgqa or self.task_hgvqa) and not self.task_vqa:
                    if hg_score > best_valid:
                        best_valid = hg_score
                        self.save("BEST")
                else:
                    if valid_score > best_valid:
                        best_valid = valid_score
                        self.save("BEST")
                if args.merge_data or args.merge_all:
                    _, _ = self.evaluate_all(eval_tuple)

                log_str += "Task %s - Epoch %d: Valid %0.2f \t HG %0.2f\n" % (task, epoch, valid_score * 100., hg_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        hg_quesid2ans = {}
        actions_class_acc = []
        rel_class_acc = []
        results = []
        hg_results = []
        activations = []

        for i, datum_tuple in enumerate(loader):
            ques_id, vid_id, feats, boxes, sent, rel_triplets, lengths, act_tokens, hg_mask, qa0, qa1, qa2, qa3, act_lengths = datum_tuple[:14]
            features = convert_sents_to_features(
                sent, self.max_seq_length, self.tokenizer)

            #process relation labels
            rel_features = convert_relations_to_features(rel_triplets, num_situations=self.num_situations,
                                                         num_rel=self.num_rel, lengths=lengths,
                                                         loss_hg_per_frame=args.loss_hg_per_frame)
            rel_feat_tgt_mask = torch.tensor(generate_rel_target_mask(num_situations=self.num_situations,
                                                                      num_rel=self.num_rel))

            rel_input_ids = torch.tensor(np.array([f.input_ids for f in rel_features]), dtype=torch.long)
            rel_segment_ids = torch.tensor(np.array([f.segment_ids for f in rel_features]), dtype=torch.long)

            if args.loss_hg_per_frame:
                tgts = [{"labels": f.targets} for f in rel_features]
            else:
                tgts = [{"labels": torch.tensor(f.targets)} for f in rel_features]

            # process action labels
            act_features = convert_relations_to_features(act_tokens, num_rel=self.num_act,
                                                         num_situations=self.num_situations, lengths=act_lengths,
                                                         loss_hg_per_frame=args.loss_hg_per_frame)
            act_feat_tgt_mask = torch.tensor(
                generate_rel_target_mask(num_situations=self.num_situations, num_rel=self.num_act))

            act_input_ids = torch.tensor(np.array([f.input_ids for f in act_features]), dtype=torch.long)
            act_segment_ids = torch.tensor(np.array([f.segment_ids for f in act_features]), dtype=torch.long)
            if args.loss_hg_per_frame:
                act_tgts = [{"labels": f.targets} for f in act_features]
            else:
                act_tgts = [{"labels": torch.tensor(f.targets)} for f in act_features]

            #process question features
            input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
            input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
            segment_ids = torch.tensor(np.array([f.segment_ids for f in features]), dtype=torch.long)

            attention = []

            with torch.no_grad():
                if torch.cuda.is_available():
                    feats, boxes = feats.cuda(), boxes.cuda()
                    input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                    rel_input_ids, rel_segment_ids, rel_feat_tgt_mask = rel_input_ids.cuda(), rel_segment_ids.cuda(), rel_feat_tgt_mask.cuda()
                    hg_mask = hg_mask.cuda()

                    if args.loss_hg_per_frame:
                        tgts = [{"labels": [g.cuda() for g in f.targets]} for f in rel_features]
                    else:
                        tgts = [{"labels": torch.tensor(f.targets).cuda()} for f in rel_features]

                    act_input_ids, act_segment_ids, act_feat_tgt_mask = act_input_ids.cuda(), act_segment_ids.cuda(), act_feat_tgt_mask.cuda()

                    if args.loss_hg_per_frame:
                        act_tgts = [{"labels": [g.cuda() for g in f.targets]} for f in act_features]
                    else:
                        act_tgts = [{"labels": torch.tensor(f.targets).cuda()} for f in act_features]

                if args.gt_hg:
                    rel_tgt_ids = rel_input_ids
                    act_tgt_ids = act_input_ids
                else:
                    rel_tgt_ids, act_tgt_ids = None, None

                logit, rel_logit, act_logit, hg_logit, attn_probs  = self.model(feats, boxes, input_ids=input_ids, input_masks=input_mask,
                                                           segment_ids=segment_ids, rel_segment_ids=rel_segment_ids,
                                                           rel_tgt_mask=rel_feat_tgt_mask, act_segment_ids=act_segment_ids,
                                                          act_tgt_mask=act_feat_tgt_mask, hg_mask=hg_mask,
                                                        rel_tgt_ids=rel_tgt_ids, act_tgt_ids=act_tgt_ids )

                b = logit.size(0)
                indices = self.matcher({'pred_logits': rel_logit}, tgts)
                rel_lbls, rel_acc = self.get_target_classes({'pred_logits': rel_logit}, tgts, indices,
                                            empty_weight=self.empty_weight, loss_hg_per_frame=args.loss_hg_per_frame)
                rel_class_acc.append(rel_acc)
                # compute hungarian matching loss between predicted actions and target actions
                # Retrieve the matching between the outputs of the last layer and the targets
                act_indices = self.matcher({'pred_logits': act_logit}, act_tgts)
                act_lbls, act_acc = self.get_target_classes({'pred_logits': act_logit}, act_tgts, act_indices,
                                            empty_weight=self.empty_weight_acts, loss_hg_per_frame=args.loss_hg_per_frame)

                actions_class_acc.append(act_acc)
                if args.output_attention:
                    last_layer_att_score = torch.squeeze(attn_probs[attn_idx[args.cross_attn_type]][-1]['attn'][:, :, 0, :])  # batch_size, att_head, target_num_feat, source_num_feat -> use all att head and CLS as target
                    # print(last_layer_att_score.shape)
                    last_layer_att_score = last_layer_att_score.cpu().numpy().tolist()

                    # activation_score = torch.squeeze(attn_probs[-2][-1])
                    # activation_score = activation_score.cpu().numpy().tolist()

                else:
                    last_layer_att_score = []
                    activation_score = []



                score, label = logit.max(1)

                act_lbls = act_lbls.view(b, 16, -1).cpu().numpy().tolist()
                rel_lbls = rel_lbls.view(b, 16, -1).cpu().numpy().tolist()
                for qid, l, act_l, act_gt, rel_l, rel_gt in zip(ques_id, label.cpu().numpy(), act_lbls, act_tokens.numpy().tolist(),
                                                                rel_lbls, rel_triplets.numpy().tolist()):
                    quesid2ans[qid] = int(l)
                    results.append(
                        {
                            "questionId": qid,
                            "prediction": l.tolist(),
                            "attention": last_layer_att_score,
                            "act_gt": act_gt,
                            "act_pred": act_l,
                            "rel_gt": rel_gt,
                            "rel_pred":rel_l
                        }
                    )


                hg_score, hg_label = hg_logit.max(1)
                for qid, l in zip(ques_id, hg_label.cpu().numpy()):

                    hg_quesid2ans[qid] = int(l)
                    hg_results.append(
                        {
                            "questionId": qid,
                            "prediction": l.tolist(),
                            "attention": last_layer_att_score
                        }
                    )

                    # activations.append(
                    #     {
                    #         "questionId": qid,
                    #         "prediction": l,
                    #         "activation": activation_score
                    #     }
                    # )

        exp_name = args.output.split('/')[-1]
        try:
            fpath = 'snap/star/{output}'.format(output=exp_name)
            os.makedirs(fpath, exist_ok=True)
            evaluator.save_json(results, f'{fpath}/val_attentions_cross_2.json')
            evaluator.save_json(hg_results, f'{fpath}/hg_val_attentions_cross_2.json')
        except:
            print('some error happened in saving predictions..')
        # evaluator.save_json(results, 'snap/star/{output}/activations.json'.format(output=exp_name))
        print('Relation Class Acc: %0.4f \t Action Class Acc: %0.4f'%(sum(rel_class_acc)/len(rel_class_acc),
                                                                        sum(actions_class_acc)/len(actions_class_acc)))
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
            dump2 = os.path.join("/".join(dump.split("/")[:-1]), "predict_hg.json")
            evaluator.dump_result(hg_quesid2ans, dump2)
        return quesid2ans, hg_quesid2ans


    def test(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        hg_quesid2ans = {}
        results = []
        hg_results = []
        activations = []

        for i, datum_tuple in enumerate(loader):
            ques_id, vid_id, feats, boxes, sent, rel_triplets, lengths, act_tokens, hg_mask, qa0, qa1, qa2, qa3, act_lengths = datum_tuple[:14]
            features = convert_sents_to_features(
                sent, self.max_seq_length, self.tokenizer)

            bsz=len(ques_id)

            #process relation labels
            rel_features = convert_relations_to_features_test(rel_triplets, num_situations=self.num_situations,
                                                         num_rel=self.num_rel, lengths=lengths, bsize=bsz)
            rel_feat_tgt_mask = torch.tensor(generate_rel_target_mask(num_situations=self.num_situations,
                                                                      num_rel=self.num_rel))

            rel_segment_ids = torch.tensor(np.array([f.segment_ids for f in rel_features]), dtype=torch.long)
        

            # process action labels
            act_features = convert_relations_to_features_test(act_tokens, num_rel=self.num_act,
                                                         num_situations=self.num_situations, lengths=act_lengths, bsize=bsz)
            act_feat_tgt_mask = torch.tensor(
                generate_rel_target_mask(num_situations=self.num_situations, num_rel=self.num_act))

            act_segment_ids = torch.tensor(np.array([f.segment_ids for f in act_features]), dtype=torch.long)
            
            #process question features
            input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
            input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
            segment_ids = torch.tensor(np.array([f.segment_ids for f in features]), dtype=torch.long)

            attention = []

            with torch.no_grad():
                if torch.cuda.is_available():
                    feats, boxes = feats.cuda(), boxes.cuda()
                    input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                    rel_segment_ids, rel_feat_tgt_mask = rel_segment_ids.cuda(), rel_feat_tgt_mask.cuda()
                    hg_mask = hg_mask.cuda()

                    act_segment_ids, act_feat_tgt_mask = act_segment_ids.cuda(), act_feat_tgt_mask.cuda()


                rel_tgt_ids, act_tgt_ids = None, None

                logit, rel_logit, act_logit, hg_logit, attn_probs  = self.model(feats, boxes, input_ids=input_ids, input_masks=input_mask,
                                                           segment_ids=segment_ids, rel_segment_ids=rel_segment_ids,
                                                           rel_tgt_mask=rel_feat_tgt_mask, act_segment_ids=act_segment_ids,
                                                          act_tgt_mask=act_feat_tgt_mask, hg_mask=hg_mask,
                                                        rel_tgt_ids=rel_tgt_ids, act_tgt_ids=act_tgt_ids )


                if args.output_attention:
                    last_layer_att_score = torch.squeeze(attn_probs[attn_idx[args.cross_attn_type]][-1]['attn'][:, :, 0, :])  # batch_size, att_head, target_num_feat, source_num_feat -> use all att head and CLS as target
                    # print(last_layer_att_score.shape)
                    last_layer_att_score = last_layer_att_score.cpu().numpy().tolist()

                    # activation_score = torch.squeeze(attn_probs[-2][-1])
                    # activation_score = activation_score.cpu().numpy().tolist()

                else:
                    last_layer_att_score = []
                    activation_score = []



                score, label = logit.max(1)
                b = feats.size(0)


                for qid, l in zip(ques_id, label.cpu().numpy()):
                    # ans = dset.label2ans[l]
                    quesid2ans[qid] = int(l)
                    results.append(
                        {
                            "questionId": qid,
                            "prediction": l.tolist(),
                            "attention": last_layer_att_score,
                            "act_gt": [],
                            "act_pred": [],
                            "rel_gt": [],
                            "rel_pred":[]
                        }
                    )
                hg_score, hg_label = hg_logit.max(1)
                for qid, l in zip(ques_id, hg_label.cpu().numpy()):
                    # ans = dset.label2ans[l]
                    hg_quesid2ans[qid] = int(l)
                    hg_results.append(
                        {
                            "questionId": qid,
                            "prediction": l.tolist(),
                            "attention": last_layer_att_score
                        }
                    )

                    # activations.append(
                    #     {
                    #         "questionId": qid,
                    #         "prediction": l,
                    #         "activation": activation_score
                    #     }
                    # )

        exp_name = args.output.split('/')[-1]
        try:
            fpath = '/media/sfarkya/8AD44225D442143B/STAR/snap/star/{output}'.format(output=exp_name)
            os.makedirs(fpath, exist_ok=True)
            evaluator.save_json(results, f'{fpath}/val_attentions_cross_2.json')
            evaluator.save_json(hg_results, f'{fpath}/hg_val_attentions_cross_2.json')
        except:
            print('some error happened in saving predictions..')

        # evaluator.save_json(results, 'snap/star/{output}/activations.json'.format(output=exp_name))

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
            dump2 = os.path.join("/".join(dump.split("/")[:-1]), "predict_hg.json")
            evaluator.dump_result(hg_quesid2ans, dump2)
        return quesid2ans, hg_quesid2ans


    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans, hg_quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans), evaluator.evaluate(hg_quesid2ans)

    def evaluate_all(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans, hg_quesid2ans = self.predict(eval_tuple, dump)

        quesid2ans_sorted = self.sort_by_qtype(quesid2ans)
        q2ans = [f'{qtype} Acc:{evaluator.evaluate(qtype2res):9.4f}' for qtype, qtype2res in quesid2ans_sorted.items()]
        print(" ".join(q2ans))

        hg_quesid2ans_sorted = self.sort_by_qtype(hg_quesid2ans)
        q2ans = [f'{qtype} Acc:{evaluator.evaluate(qtype2res):9.4f}' for qtype, qtype2res in hg_quesid2ans_sorted.items()]
        print(" ".join(q2ans))

        return evaluator.evaluate(quesid2ans), evaluator.evaluate(hg_quesid2ans)

    def sort_by_qtype(self, quesid2ans):
        preds_by_qtype = {
            'Interaction': {},
            'Sequence': {},
            'Prediction': {},
            'Feasibility': {}
        }
        for qid, ans in quesid2ans.items():
            if qid.startswith('Interaction'):
                preds_by_qtype['Interaction'][qid] = ans
            elif qid.startswith('Sequence'):
                preds_by_qtype['Sequence'][qid] = ans
            elif qid.startswith('Prediction'):
                preds_by_qtype['Prediction'][qid] = ans
            elif qid.startswith('Feasibility'):
                preds_by_qtype['Feasibility'][qid] = ans
        return preds_by_qtype

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, vid_id, feats, boxes, sent, rel_triplets, lengths, act_tokens, hg_mask,
                qa0, qa1, qa2, qa3, act_lengths, target) in enumerate(loader): #(ques_id, feats, boxes, sent, target)

            _, label = target.max(dim=1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                quesid2ans[qid] = int(l)
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path, map_location=torch.device(device))
        if not args.train:
            for key in list(state_dict.keys()):
                if 'module.' in key:
                    state_dict[key.replace('module.', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Build Class
    star = STAR()


    # Load Model
    if args.load is not None:
        star.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            star.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'test' in args.test:
            result = star.test(
                get_tuple('test', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
            print(result)
        if 'valid' in args.test:
            result = star.evaluate(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'valid_predict.json')
            )
            print(result)
    else:
        print('Splits in Train data:', star.train_tuple.dataset.splits)
        if star.valid_tuple is not None:
            print('Splits in Valid data:', star.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (star.oracle_score(star.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        star.train(star.train_tuple, star.valid_tuple)


