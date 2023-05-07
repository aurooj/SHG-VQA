# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import time
import numpy as np
import gc
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn


# from src.lxrt.detr import SetCriterion
from src.param import args
from src.lxrt.tokenization import BertTokenizer
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.agqa_model import AGQAModel
from src.visualization_tools.vis_utils import accuracy
from src.lxrt.entry import convert_sents_to_features, convert_relations_to_features, generate_rel_target_mask, convert_relations_to_features_test
from src.lxrt.matcher import HungarianMatcher
# from src.lxrt.detr import SetCriterion
if args.patches:
    raise NotImplementedError
else:
    from src.tasks.agqa_data import AGQADataset, AGQATorchDataset, AGQAEvaluator

print(args, flush=True)
#dictionary to hold indices to extract attention from based on the cross attn type
attn_idx = {
    'cross': 2,
    'cross_self': 4,
    'old': 2,
    'self': 4
}
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = AGQADataset(splits)
    tset = AGQATorchDataset(dset)
    evaluator = AGQAEvaluator(dset)

    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        # collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )


    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class AGQA:
    def __init__(self):
        self.task_vqa = args.task_vqa
        self.task_vhga = args.task_vhga
        self.task_hgqa = args.task_hgqa
        self.task_q = args.task_q
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



        self.model = AGQAModel(self.train_tuple.dataset.num_answers, num_queries=128,
                               num_actions=len(self.train_tuple.dataset.action_classes))
        self.background_idx = 0
        self.clip_len = args.CLIP_LEN
        self.num_situations = self.train_tuple.dataset.num_situations
        self.num_rel = self.train_tuple.dataset.num_rel
        self.num_actions = len(self.train_tuple.dataset.action_classes)
        self.num_act = self.train_tuple.dataset.num_act

        self.max_seq_length = self.model.max_seq_length
        self.rel_classes = 456  # todo: change rel classes
        self.eos_coef_rel = 0.1
        empty_weight = torch.ones(self.rel_classes + 1)
        empty_weight[
            self.background_idx] = self.eos_coef_rel  # ideally should be set for the empty class index (0 in our case)
        self.empty_weight = empty_weight

        self.eos_coef_acts = 0.1
        empty_weight_acts = torch.ones(self.num_actions + 1)
        empty_weight_acts[self.background_idx] = self.eos_coef_acts
        self.empty_weight_acts = empty_weight_acts



        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        # if args.load_lxmert_qa is not None:
        #     load_lxmert_qa(args.load_lxmert_qa, self.model,
        #                    label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        if args.test == None and args.load == None:
            if torch.cuda.is_available():
                if args.multiGPU:
                    self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

                self.model.to(device)


        #relation loss
        self.matcher = HungarianMatcher(cost_class=1, loss_hg_per_frame=args.loss_hg_per_frame, clip_len=self.clip_len)
        self.weight_dict = {"loss_ce": 1, "loss_bbox": 0}
        self.weight_dict["loss_giou"] = 0

        losses = ["labels"]
        # self.criterion = SetCriterion(
        #     self.num_classes, matcher=self.matcher, weight_dict=weight_dict, eos_coef=args.no_object_weight, losses=losses,
        # )
        # self.criterion.to(self.device)

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

        # self.scheduler = MultiStepLR(self.optim, milestones=[15, 35], gamma=0.9)
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

    def get_target_classes(self, outputs, targets, indices, empty_weight, log=True, loss_hg_per_frame=False):
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
            assert num_queries % self.clip_len == 0
            src_logits = src_logits.reshape(b * self.clip_len, num_queries // self.clip_len, -1)
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
            assert num_queries % self.clip_len == 0
            src_logits = src_logits.reshape(b * self.clip_len, num_queries // self.clip_len, -1)
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
        if args.train_sub_set:
            SET_ITERATIONS = len(loader)
        else:
            SET_ITERATIONS = len(loader)    # Change for Fixed set of QAs for each epoch


        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        early_stopping_index = 0

        if self.valid_tuple is not None:
            STOP_AFTER = 10
        else:
            STOP_AFTER = args.epochs


        for epoch in range(args.epochs):
            quesid2ans = {}

            if early_stopping_index < STOP_AFTER:  # if validation hasn't changed for 10 epochs, stop

                for i, (ques_id, vid_id, feats, boxes, sent, rel_triplets, lengths, act_tokens, hg_mask,
                         act_lengths, target) in iter_wrapper(enumerate(loader)):

                    # if i < (SET_ITERATIONS/args.batch_size):

                    self.model.train()
                    self.model.module.vid_encoder.eval()
                    self.optim.zero_grad(set_to_none=True)

                    train_features = convert_sents_to_features(
                        sent, self.max_seq_length, self.tokenizer)

                    # process relation triplets
                    rel_features = convert_relations_to_features(rel_triplets, num_rel=self.num_rel,
                                                                 num_situations=self.num_situations, lengths=lengths,
                                                                 loss_hg_per_frame=args.loss_hg_per_frame)

                    rel_feat_tgt_mask = torch.as_tensor(
                        generate_rel_target_mask(num_situations=self.num_situations, num_rel=self.num_rel))

                    rel_input_ids = torch.as_tensor(np.array([f.input_ids for f in rel_features]), dtype=torch.long)
                    rel_segment_ids = torch.as_tensor(np.array([f.segment_ids for f in rel_features]), dtype=torch.long)
                    if args.loss_hg_per_frame:
                        tgts = [{"labels": f.targets} for f in rel_features]
                    else:
                        tgts = [{"labels": torch.tensor(f.targets)} for f in rel_features]

                    # process action labels
                    act_features = convert_relations_to_features(act_tokens, num_rel=self.num_act,
                                                                 num_situations=self.num_situations,
                                                                 lengths=act_lengths,
                                                                 loss_hg_per_frame=args.loss_hg_per_frame)
                    act_feat_tgt_mask = torch.as_tensor(
                        generate_rel_target_mask(num_situations=self.num_situations, num_rel=self.num_act))

                    act_input_ids = torch.as_tensor(np.array([f.input_ids for f in act_features]), dtype=torch.long)
                    act_segment_ids = torch.as_tensor(np.array([f.segment_ids for f in act_features]), dtype=torch.long)
                    if args.loss_hg_per_frame:
                        act_tgts = [{"labels": f.targets} for f in act_features]
                    else:
                        act_tgts = [{"labels": torch.tensor(f.targets)} for f in act_features]

                    # process question features
                    input_ids = torch.as_tensor(np.array([f.input_ids for f in train_features]), dtype=torch.long)
                    input_mask = torch.as_tensor(np.array([f.input_mask for f in train_features]), dtype=torch.long)
                    segment_ids = torch.as_tensor(np.array([f.segment_ids for f in train_features]), dtype=torch.long)

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

                    logit, rel_logit, act_logit, hg_logit, attn_probs = self.model(feats, boxes, input_ids=input_ids,
                                                                                   input_masks=input_mask,
                                                                                   segment_ids=segment_ids,
                                                                                   rel_segment_ids=rel_segment_ids,
                                                                                   rel_tgt_mask=rel_feat_tgt_mask,
                                                                                   act_segment_ids=act_segment_ids,
                                                                                   act_tgt_mask=act_feat_tgt_mask,
                                                                                   hg_mask=hg_mask,
                                                                                   rel_tgt_ids=rel_tgt_ids,
                                                                                   act_tgt_ids=act_tgt_ids)

                    assert logit.dim() == target.dim() == 2
                    losses = ()
                    total_loss = 0.
                    vqa_loss_str = ""
                    hgqa_loss_str = ""
                    hg_loss_str = ""

                    hgqa_loss = self.bce_loss(hg_logit, target)
                    hgqa_loss = hgqa_loss * hg_logit.size(1)
                    total_loss += hgqa_loss
                    losses += (hgqa_loss.detach(),)
                    hgqa_loss_str = "\tHGQA loss= %0.4f" % hgqa_loss.detach().item()

                    if not args.gt_hg:
                        # compute hungarian matching loss between predicted relation and target relations
                        # Retrieve the matching between the outputs of the last layer and the targets
                        indices = self.matcher({'pred_logits': rel_logit}, tgts,
                                               )
                        rel_loss = self.loss_labels({'pred_logits': rel_logit}, tgts, indices,
                                                    empty_weight=self.empty_weight,
                                                    loss_hg_per_frame=args.loss_hg_per_frame)

                        # compute hungarian matching loss between predicted actions and target actions
                        # Retrieve the matching between the outputs of the last layer and the targets
                        act_indices = self.matcher({'pred_logits': act_logit}, act_tgts)
                        act_loss = self.loss_labels({'pred_logits': act_logit}, act_tgts, act_indices,
                                                    empty_weight=self.empty_weight_acts,
                                                    loss_hg_per_frame=args.loss_hg_per_frame)

                        # add relation loss
                        for los, loss_value in rel_loss.items():
                            if los in self.weight_dict:
                                los_w = self.weight_dict[los] * loss_value
                                total_loss += los_w
                                losses += (los_w.detach(),)

                        # add action loss
                        for los, loss_value in act_loss.items():
                            if los in self.weight_dict:
                                los_w = self.weight_dict[los] * loss_value
                                total_loss += los_w
                                losses += (los_w.detach(),)

                        hg_loss_str = "\tRel loss= %0.4f \tAct loss= %0.4f\n" \
                                      "Rel class error= %0.4f \t Act class error= %0.4f\n" % (
                                          rel_loss["loss_ce"].detach(), act_loss["loss_ce"].detach(),
                                          rel_loss["class_error"].detach(), act_loss["class_error"].detach())

                    if i % args.log_freq == 0:
                        log_loss = "\nEpoch %d: Total loss= %0.4f " % (epoch, total_loss.detach())
                        log_loss += vqa_loss_str + hgqa_loss_str + hg_loss_str
                        print(log_loss, flush=True)

                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    self.optim.step()

                    score, label = hg_logit.max(1)
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        # ans = dset.label2ans[l]
                        ans = l
                        quesid2ans[qid] = ans



                log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluateOverall(quesid2ans) * 100.)
                # if (self.valid_tuple == None):
                #     self.save("CURRENT")
                self.save("CURRENT")

                # to handle GPU OOM error
                if torch.cuda.is_available():
                    # gc.collect()
                    torch.cuda.empty_cache()


                if self.valid_tuple is not None:  # Do Validation
                    if not self.task_q and not self.task_vqa:
                        valid_score, hg_score = self.evaluate(eval_tuple) #todo: keep best model based on hg_score
                        if (self.task_hgqa or self.task_vhga) and not self.task_vqa:
                            if hg_score > best_valid:
                                best_valid = hg_score
                                self.save("BEST")
                        log_str += "Epoch %d: Valid %0.2f \t HG %0.2f\n" % (epoch, valid_score * 100., hg_score * 100.) + \
                                   "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

                    else:
                        valid_score = self.evaluate(eval_tuple)
                        if valid_score > best_valid:
                            best_valid = valid_score
                            self.save("BEST")

                        log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                                   "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)


                print(log_str, end='', flush=True)

                if self.valid_tuple is not None:
                    if (hg_score < best_valid):
                        early_stopping_index += 1
                    else:
                        early_stopping_index = 0

                else:
                    early_stopping_index += 1


            else:
                print("Stopping Early...", flush=True)
                break

            # with open(self.output + "/log.log", 'a') as f:
            #     f.write(log_str)
            #     f.flush()
            # self.scheduler.step()


        self.save("LAST")



    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        hg_quesid2ans = {}
        results = []
        hg_results = []
        actions_class_acc = []
        rel_class_acc = []
        activations = []

        # iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        for i, datum_tuple in (enumerate(loader)):
            ques_id, vid_id, feats, boxes, sent, rel_triplets, lengths, act_tokens, hg_mask, act_lengths = datum_tuple[:10]         # todo: changed to 10

            features = convert_sents_to_features(
                sent, self.max_seq_length, self.tokenizer)

            # process relation labels
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

            # process question features
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

                logit, rel_logit, act_logit, hg_logit, attn_probs = self.model(feats, boxes, input_ids=input_ids,
                                                                               input_masks=input_mask,
                                                                               segment_ids=segment_ids,
                                                                               rel_segment_ids=rel_segment_ids,
                                                                               rel_tgt_mask=rel_feat_tgt_mask,
                                                                               act_segment_ids=act_segment_ids,
                                                                               act_tgt_mask=act_feat_tgt_mask,
                                                                               hg_mask=hg_mask,
                                                                               rel_tgt_ids=rel_tgt_ids,
                                                                               act_tgt_ids=act_tgt_ids)
                b = logit.size(0)
                indices = self.matcher({'pred_logits': rel_logit}, tgts,
                                       )
                rel_lbls, rel_acc = self.get_target_classes({'pred_logits': rel_logit}, tgts, indices,
                                                            empty_weight=self.empty_weight,
                                                            loss_hg_per_frame=args.loss_hg_per_frame)
                rel_class_acc.append(rel_acc)
                # compute hungarian matching loss between predicted actions and target actions
                # Retrieve the matching between the outputs of the last layer and the targets
                act_indices = self.matcher({'pred_logits': act_logit}, act_tgts)
                act_lbls, act_acc = self.get_target_classes({'pred_logits': act_logit}, act_tgts, act_indices,
                                                            empty_weight=self.empty_weight_acts,
                                                            loss_hg_per_frame=args.loss_hg_per_frame)

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
                    # ans = dset.label2ans[l]
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


        # exp_name = args.output.split('/')[-1]
        # evaluator.save_json(results, 'snap/agqa/{output}/val_attentions_cross_2.json'.format(output=exp_name))
        # evaluator.save_json(hg_results, 'snap/agqa/{output}/hg_val_attentions_cross_2.json'.format(output=exp_name))
        # evaluator.save_json(results, 'snap/star/{output}/activations.json'.format(output=exp_name))

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
            dump2 = os.path.join("/".join(dump.split("/")[:-1]), "predict_hg.json")
            evaluator.dump_result(hg_quesid2ans, dump2)

        if self.task_q or self.task_vqa:
            return quesid2ans
        else:
            return quesid2ans, hg_quesid2ans


    def test(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        hg_quesid2ans = {}
        results = []
        hg_results = []
        activations = []
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        for i, datum_tuple in iter_wrapper(enumerate(loader)):
            ques_id, vid_id, feats, boxes, sent, rel_triplets, lengths, act_tokens, hg_mask, act_lengths = datum_tuple[:10]
            features = convert_sents_to_features(
                sent, self.max_seq_length, self.tokenizer)


            bsz= len(features)

            #process relation labels

            if not self.task_q:
                rel_features = convert_relations_to_features_test(rel_triplets, num_situations=self.num_situations,
                                                             num_rel=self.num_rel, lengths=lengths, bsize=bsz)
                rel_feat_tgt_mask = torch.tensor(generate_rel_target_mask(num_situations=self.num_situations,
                                                                          num_rel=self.num_rel))

                # rel_input_ids = torch.tensor(np.array([f.input_ids for f in rel_features]), dtype=torch.long)
                rel_segment_ids = torch.tensor(np.array([f.segment_ids for f in rel_features]), dtype=torch.long)
                # tgts = [{"labels": torch.tensor(f.targets)} for f in rel_features]

                # process action labels
                act_features = convert_relations_to_features_test(act_tokens, num_rel=self.num_act,
                                                             num_situations=self.num_situations, lengths=act_lengths, bsize=bsz)
                act_feat_tgt_mask = torch.tensor(
                    generate_rel_target_mask(num_situations=self.num_situations, num_rel=self.num_act))

                # act_input_ids = torch.tensor(np.array([f.input_ids for f in act_features]), dtype=torch.long)
                act_segment_ids = torch.tensor(np.array([f.segment_ids for f in act_features]), dtype=torch.long)
                # act_tgts = [{"labels": torch.tensor(f.targets)} for f in act_features]

            #process question features
            input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
            input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
            segment_ids = torch.tensor(np.array([f.segment_ids for f in features]), dtype=torch.long)

            attention = []

            with torch.no_grad():
                if torch.cuda.is_available():
                    feats, boxes = feats.cuda(), boxes.cuda()
                    input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()

                    if not self.task_q:
                        rel_segment_ids, rel_feat_tgt_mask = rel_segment_ids.cuda(), rel_feat_tgt_mask.cuda()
                        hg_mask = hg_mask.cuda()

                        # tgts = [{"labels": torch.tensor(f.targets).cuda()} for f in rel_features]

                        act_segment_ids, act_feat_tgt_mask = act_segment_ids.cuda(), act_feat_tgt_mask.cuda()
                        # act_tgts = [{"labels": torch.tensor(f.targets).cuda()} for f in act_features]

                # if args.gt_hg:
                #     rel_tgt_ids = rel_input_ids
                #     act_tgt_ids = act_input_ids
                # else:
                rel_tgt_ids, act_tgt_ids = None, None

                logit, rel_logit, act_logit, hg_logit, attn_probs  = self.model(feats, boxes, input_ids=input_ids, input_masks=input_mask,
                                                           segment_ids=segment_ids, rel_segment_ids=rel_segment_ids,
                                                           rel_tgt_mask=rel_feat_tgt_mask, act_segment_ids=act_segment_ids,
                                                          act_tgt_mask=act_feat_tgt_mask, hg_mask=hg_mask,
                                                        rel_tgt_ids=rel_tgt_ids, act_tgt_ids=act_tgt_ids )

                # indices = self.matcher({'pred_logits': rel_logit}, tgts,
                #                        )
                # rel_lbls = self.get_target_classes({'pred_logits': rel_logit}, tgts, indices,
                #                             empty_weight=self.empty_weight)
                #
                # # compute hungarian matching loss between predicted actions and target actions
                # # Retrieve the matching between the outputs of the last layer and the targets
                # act_indices = self.matcher({'pred_logits': act_logit}, act_tgts)
                # act_lbls = self.get_target_classes({'pred_logits': act_logit}, act_tgts, act_indices,
                #                             empty_weight=self.empty_weight_acts)

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
                # b = feats.size(0)
                #todo: save predicted hypergraph for test set, note: will be without permutations for bipartite matching

                # act_lbls = act_lbls.view(b, 16, -1).cpu().numpy().tolist()
                # rel_lbls = rel_lbls.view(b, 16, -1).cpu().numpy().tolist()
                # for qid, l, act_l, act_gt, rel_l, rel_gt in zip(ques_id, label.cpu().numpy(), act_lbls, act_tokens.numpy().tolist(),
                #                                                 rel_lbls, rel_triplets.numpy().tolist()):
                #     # ans = dset.label2ans[l]
                #     quesid2ans[qid] = int(l)
                #     results.append(
                #         {
                #             "questionId": qid,
                #             "prediction": l.tolist(),
                #             "attention": last_layer_att_score,
                #             "act_gt": act_gt,
                #             "act_pred": act_l,
                #             "rel_gt": rel_gt,
                #             "rel_pred":rel_l
                #         }
                #     )

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

        # exp_name = args.output.split('/')[-2:]
        # evaluator.save_json(results, 'snap/agqa/{output}/val_attentions_cross_2.json'.format(output='/'.join(exp_name)))
        # evaluator.save_json(hg_results, 'snap/agqa/{output}/hg_val_attentions_cross_2.json'.format(output='/'.join(exp_name)))
        # evaluator.save_json(results, 'snap/star/{output}/activations.json'.format(output=exp_name))

        if dump is not None:
            # evaluator.dump_result(quesid2ans, dump)
            # dump2 = os.path.join("/".join(dump.split("/")[:-1]), "predict_hg.json")
            evaluator.dump_result(hg_quesid2ans, dump)
        return hg_quesid2ans


    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans, hg_quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluateOverall(quesid2ans), evaluator.evaluateOverall(hg_quesid2ans)



    def evaluateAllQtypes(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        hg_quesid2ans = self.test(eval_tuple, dump)
        return evaluator.evaluateAllQtypes(hg_quesid2ans)




    def evaluateTestSplits(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        hg_quesid2ans = self.test(eval_tuple, dump)

        if args.indirect_ref:
            all = evaluator.evaluateAllQtypes(hg_quesid2ans)
            indirect, precisionQs = evaluator.evaluateIndirectRef(hg_quesid2ans)
            precisionScore = evaluator.evaluatePrecision(precisionQs)

            return all, indirect, precisionScore


        elif args.novel_comp:
            return evaluator.evaluateNovelComp(hg_quesid2ans)


        elif args.comp_steps:
            return evaluator.evaluateCompSteps(hg_quesid2ans)

        else:
            return evaluator.evaluateAllQtypes(hg_quesid2ans)





    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, vid_id, feats, boxes, sent, rel_triplets, lengths, act_tokens, hg_mask,
                act_lengths, target) in enumerate(loader): #(ques_id, feats, boxes, sent, target)
            # _, label = datum['target'].max(1)
            _, label = target.max(1)

            for qid, l in zip(ques_id, label.cpu().numpy()):
                # print(qid, l)
                # ans = dset.label2ans[l]
                quesid2ans[qid] = int(l)

        # print('Done', flush=True)
        return evaluator.evaluate(quesid2ans)


    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))


    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path, map_location=torch.device(device))
        for key in list(state_dict.keys()):
            if 'module.' in key:
                state_dict[key.replace('module.', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=True)
        if torch.cuda.is_available():
            if args.multiGPU:
                self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
            self.model.to(device)


if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.empty_cache()

    # Build Class
    agqa = AGQA()


    # Load Model
    if args.load is not None:
        agqa.load(args.load)


    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test

        if 'valid' in args.test:
            result, result2 = agqa.evaluateAllQtypes(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=None
            )

            log_str2 = "Valid HQ results:\nOverall %0.2f\n" % (result2[0] * 100.) + \
                       "object-relationship: %0.2f\n" % (result2[1] * 100.) + \
                       "relationship-action: %0.2f\n" % (result2[2] * 100.) + \
                       "object-action: %0.2f\n" % (result2[3] * 100.) + \
                       "superlative: %0.2f\n" % (result2[4] * 100.) + \
                       "sequencing: %0.2f\n" % (result2[5] * 100.) + \
                       "exists: %0.2f\n" % (result2[6] * 100.) + \
                       "duration-comparison: %0.2f\n" % (result2[7] * 100.) + \
                       "action-recognition: %0.2f\n" % (result2[8] * 100.)

            print(log_str2, flush=True)



        if 'test' in args.test:
            testData = get_tuple('test', bs=args.batch_size,
                                 shuffle=False, drop_last=False)

            if args.indirect_ref:
                result, result2, result3 = agqa.evaluateTestSplits(testData,
                                                                   dump=os.path.join(args.output, 'indirect_refs.json'))

                log_str = "\nTest Results:\n Overall %0.2f\n" % (result[0] * 100.) + \
                          "binary: %0.2f\n" % (result[1] * 100.) + \
                          "open: %0.2f\n" % (result[2] * 100.) + \
                          "object-relationship: %0.2f\n" % (result[3] * 100.) + \
                          "object-relationship binary: %0.2f\n" % (result[4] * 100.) + \
                          "object-relationship open: %0.2f\n" % (result[5] * 100.) + \
                          "relationship-action: %0.2f\n" % (result[6] * 100.) + \
                          "object-action: %0.2f\n" % (result[7] * 100.) + \
                          "superlative: %0.2f\n" % (result[8] * 100.) + \
                          "superlative binary: %0.2f\n" % (result[9] * 100.) + \
                          "superlative open: %0.2f\n" % (result[10] * 100.) + \
                          "sequencing: %0.2f\n" % (result[11] * 100.) + \
                          "sequencing binary: %0.2f\n" % (result[12] * 100.) + \
                          "sequencing open: %0.2f\n" % (result[13] * 100.) + \
                          "exists: %0.2f\n" % (result[14] * 100.) + \
                          "duration-comparison: %0.2f\n" % (result[15] * 100.) + \
                          "duration-comparison binary: %0.2f\n" % (result[16] * 100.) + \
                          "duration-comparison open: %0.2f\n" % (result[17] * 100.) + \
                          "action-recognition: %0.2f\n" % (result[18] * 100.) + \
                          "\nSEMANTICS:\n" + \
                          "object: %0.2f\n" % (result[19] * 100.) + \
                          "object binary: %0.2f\n" % (result[20] * 100.) + \
                          "object open: %0.2f\n" % (result[21] * 100.) + \
                          "relationship: %0.2f\n" % (result[22] * 100.) + \
                          "action: %0.2f\n" % (result[23] * 100.) + \
                          "action binary: %0.2f\n" % (result[24] * 100.) + \
                          "action open: %0.2f\n" % (result[25] * 100.) + \
                          "\nSTRUCTURE:\n" + \
                          "query: %0.2f\n" % (result[26] * 100.) + \
                          "compare: %0.2f\n" % (result[27] * 100.) + \
                          "choose: %0.2f\n" % (result[28] * 100.) + \
                          "logic: %0.2f\n" % (result[29] * 100.) + \
                          "verify: %0.2f\n" % (result[30] * 100.)

                log_str2 = "\nTest Indirect References:\n" \
                           "object recall: %0.2f\n" % (result2[0] * 100.) + \
                           "object recall binary: %0.2f\n" % (result2[1] * 100.) + \
                           "object recall open: %0.2f\n" % (result2[2] * 100.) + \
                           "action recall: %0.2f\n" % (result2[3] * 100.) + \
                           "action recall binary: %0.2f\n" % (result2[4] * 100.) + \
                           "action recall open: %0.2f\n" % (result2[5] * 100.) + \
                           "localization recall: %0.2f\n" % (result2[6] * 100.) + \
                           "localization recall binary: %0.2f\n" % (result2[7] * 100.) + \
                           "localization recall open: %0.2f\n" % (result2[8] * 100.)

                log_str3 = "\nTest Precision:\n" \
                           "object precision: %0.2f\n" % (result3[0] * 100.) + \
                           "object precision binary: %0.2f\n" % (result3[1] * 100.) + \
                           "object precision open: %0.2f\n" % (result3[2] * 100.) + \
                           "action precision: %0.2f\n" % (result3[3] * 100.) + \
                           "action precision binary: %0.2f\n" % (result3[4] * 100.) + \
                           "action precision open: %0.2f\n" % (result3[5] * 100.) + \
                           "localization precision: %0.2f\n" % (result3[6] * 100.) + \
                           "localization precision binary: %0.2f\n" % (result3[7] * 100.) + \
                           "localization precision open: %0.2f\n" % (result3[8] * 100.)

                print(log_str, flush=True)
                print(log_str2, flush=True)
                print(log_str3, flush=True)





            elif args.novel_comp:
                result = agqa.evaluateTestSplits(testData, dump=os.path.join(args.output, 'novel_comp.json'))

                log_str = "\nTest Novel Compositions:\n" \
                          "overall: %0.2f\n" % (result[0] * 100.) + \
                          "overall binary: %0.2f\n" % (result[1] * 100.) + \
                          "overall open: %0.2f\n" % (result[2] * 100.) + \
                          "sequencing: %0.2f\n" % (result[3] * 100.) + \
                          "sequencing binary: %0.2f\n" % (result[4] * 100.) + \
                          "sequencing open: %0.2f\n" % (result[5] * 100.) + \
                          "superlative: %0.2f\n" % (result[6] * 100.) + \
                          "superlative binary: %0.2f\n" % (result[7] * 100.) + \
                          "superlative open: %0.2f\n" % (result[8] * 100.) + \
                          "duration: %0.2f\n" % (result[9] * 100.) + \
                          "duration binary: %0.2f\n" % (result[10] * 100.) + \
                          "duration open: %0.2f\n" % (result[11] * 100.) + \
                          "object relationship: %0.2f\n" % (result[12] * 100.) + \
                          "object relationship binary: %0.2f\n" % (result[13] * 100.) + \
                          "object relationship open: %0.2f\n" % (result[14] * 100.)

                print(log_str, flush=True)


            elif args.comp_steps:
                result = agqa.evaluateTestSplits(testData, dump=os.path.join(args.output, 'comp_steps.json'))

                log_str = "\nTest Novel Compositions:\n" \
                          "overall: %0.2f\n" % (result[0] * 100.) + \
                          "overall binary: %0.2f\n" % (result[1] * 100.) + \
                          "overall open: %0.2f\n" % (result[2] * 100.)

                print(log_str, flush=True)





            else:
                result = agqa.evaluateAllQtypes(testData, dump=os.path.join(args.output, 'test_data.json'))

                log_str = "\nTest:\n Overall %0.2f\n" % (result[0] * 100.) + \
                          "binary: %0.2f\n" % (result[1] * 100.) + \
                          "open: %0.2f\n" % (result[2] * 100.) + \
                          "object-relationship: %0.2f\n" % (result[3] * 100.) + \
                          "object-relationship binary: %0.2f\n" % (result[4] * 100.) + \
                          "object-relationship open: %0.2f\n" % (result[5] * 100.) + \
                          "relationship-action: %0.2f\n" % (result[6] * 100.) + \
                          "object-action: %0.2f\n" % (result[7] * 100.) + \
                          "superlative: %0.2f\n" % (result[8] * 100.) + \
                          "superlative binary: %0.2f\n" % (result[9] * 100.) + \
                          "superlative open: %0.2f\n" % (result[10] * 100.) + \
                          "sequencing: %0.2f\n" % (result[11] * 100.) + \
                          "sequencing binary: %0.2f\n" % (result[12] * 100.) + \
                          "sequencing open: %0.2f\n" % (result[13] * 100.) + \
                          "exists: %0.2f\n" % (result[14] * 100.) + \
                          "duration-comparison: %0.2f\n" % (result[15] * 100.) + \
                          "duration-comparison binary: %0.2f\n" % (result[16] * 100.) + \
                          "duration-comparison open: %0.2f\n" % (result[17] * 100.) + \
                          "action-recognition: %0.2f\n" % (result[18] * 100.) + \
                          "\nSEMANTICS:\n" + \
                          "object: %0.2f\n" % (result[19] * 100.) + \
                          "object binary: %0.2f\n" % (result[20] * 100.) + \
                          "object open: %0.2f\n" % (result[21] * 100.) + \
                          "relationship: %0.2f\n" % (result[22] * 100.) + \
                          "action: %0.2f\n" % (result[23] * 100.) + \
                          "action binary: %0.2f\n" % (result[24] * 100.) + \
                          "action open: %0.2f\n" % (result[25] * 100.) + \
                          "\nSTRUCTURE:\n" + \
                          "query: %0.2f\n" % (result[26] * 100.) + \
                          "compare: %0.2f\n" % (result[27] * 100.) + \
                          "choose: %0.2f\n" % (result[28] * 100.) + \
                          "logic: %0.2f\n" % (result[29] * 100.) + \
                          "verify: %0.2f\n" % (result[30] * 100.)

                print(log_str, flush=True)



    else:
        #print("Train Oracle: %0.2f" % (agqa.oracle_score(agqa.train_tuple) * 100), flush=True)
        # print('Splits in Train data:', agqa.train_tuple.dataset.splits, flush=True)
        # if agqa.valid_tuple is not None:
        #     print('Splits in Valid data:', agqa.valid_tuple.dataset.splits)
        #     print("Valid Oracle: %0.2f" % (agqa.oracle_score(agqa.valid_tuple) * 100), flush=True)
        # else:
        #     print("DO NOT USE VALIDATION")

        agqa.train(agqa.train_tuple, agqa.valid_tuple)
