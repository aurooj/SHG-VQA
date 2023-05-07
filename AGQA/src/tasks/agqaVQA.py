# coding=utf-8
# Copyleft 2019 project LXRT.

# * FOR AGQA-VQA MODEL*

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
        drop_last=drop_last, pin_memory=True, prefetch_factor=2, persistent_workers=True

    )

    # , prefetch_factor=2, persistent_workers=True


    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class AGQA:
    def __init__(self):
        self.task_vqa = args.task_vqa
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

        self.model = AGQAModel(self.train_tuple.dataset.num_answers, num_queries=128,            #todo: change num of answers passed
                               num_actions=len(self.train_tuple.dataset.action_classes))


        self.max_seq_length = self.model.max_seq_length

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
                    # self.model.lxrt_encoder.multi_gpu()
                    self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

                self.model.to(device)





        #relation loss
        self.matcher = HungarianMatcher(cost_class=1)
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

    def get_target_classes(self, outputs, targets, indices, empty_weight, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = (torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])).long()
        target_classes = torch.full(src_logits.shape[:2], self.background_idx,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        return target_classes

    def loss_labels(self, outputs, targets, indices, empty_weight, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = (torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])).long()
        target_classes = torch.full(src_logits.shape[:2], 0,
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
            SET_ITERATIONS = len(loader)

        iter_wrapper = (lambda x: tqdm(x, total=SET_ITERATIONS)) if args.tqdm else (lambda x: x)

        best_valid = 0.
        early_stopping_index = 0

        for epoch in range(args.epochs):
            quesid2ans = {}

            if early_stopping_index < 10:  # if validation hasn't changed for 10 epochs, stop

                for i, (ques_id, vid_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                    if i < SET_ITERATIONS:
                        self.model.train()
                        self.optim.zero_grad(set_to_none=True)

                        train_features = convert_sents_to_features(
                            sent, self.max_seq_length, self.tokenizer)



                        #process question features
                        input_ids = torch.as_tensor(np.array([f.input_ids for f in train_features]), dtype=torch.long)
                        input_mask = torch.as_tensor(np.array([f.input_mask for f in train_features]), dtype=torch.long)
                        segment_ids = torch.as_tensor(np.array([f.segment_ids for f in train_features]), dtype=torch.long)

                        if torch.cuda.is_available():
                            feats, boxes, target = feats.to(device), boxes.to(device), target.to(device)
                            input_ids, input_mask, segment_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device)


                        if self.task_vqa:
                            logit, attn_probs = self.model(feats, boxes, input_ids=input_ids, input_masks=input_mask,
                                                                      segment_ids=segment_ids, rel_segment_ids=None,
                                                                      rel_tgt_mask=None, act_segment_ids=None,
                                                                      act_tgt_mask=None, hg_mask=None,
                                                                      rel_tgt_ids=None, act_tgt_ids=None)


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



                        if i % args.log_freq == 0:
                            log_loss = "\nEpoch %d: Total loss= %0.4f "% (epoch, total_loss.detach())
                            log_loss += vqa_loss_str + hgqa_loss_str + hg_loss_str
                            print(log_loss, flush=True)



                        total_loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                        self.optim.step()


                        if self.task_q or self.task_vqa:
                            score, label = logit.max(1)
                            for qid, l in zip(ques_id, label.cpu().numpy()):
                                # ans = dset.label2ans[l]
                                ans = l
                                quesid2ans[qid] = ans



                log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluateOverall(quesid2ans) * 100.)
                self.save("CURRENT")

                # to handle GPU OOM error
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()


                if self.valid_tuple is not None:  # Do Validation
                    valid_score = self.evaluate(eval_tuple)
                    if valid_score > best_valid:
                        best_valid = valid_score
                        self.save("BEST")

                    log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                               "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

                print(log_str, end='', flush=True)
                if valid_score < best_valid:
                    early_stopping_index += 1
                else:
                    early_stopping_index = 0


            else:
                print("Stopping Early...", flush=True)
                break



        self.save("LAST")




    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        hg_quesid2ans = {}
        results = []

        for i, datum_tuple in enumerate(loader):
            ques_id, vid_id, feats, boxes, sent, target = datum_tuple[:6]         # todo: changed to 10



            features = convert_sents_to_features(
                sent, self.max_seq_length, self.tokenizer)


            #process question features
            input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
            input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
            segment_ids = torch.tensor(np.array([f.segment_ids for f in features]), dtype=torch.long)

            attention = []

            with torch.no_grad():
                if torch.cuda.is_available():
                    feats, boxes = feats.cuda(), boxes.cuda()
                    input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()

                if self.task_vqa:
                    logit, attn_probs = self.model(feats, boxes, input_ids=input_ids, input_masks=input_mask,
                                                   segment_ids=segment_ids, rel_segment_ids=None,
                                                   rel_tgt_mask=None, act_segment_ids=None,
                                                   act_tgt_mask=None, hg_mask=None,
                                                   rel_tgt_ids=None, act_tgt_ids=None)


                if args.output_attention:
                    last_layer_att_score = torch.squeeze(attn_probs[attn_idx[args.cross_attn_type]][-1]['attn'][:, :, 0, :])  # batch_size, att_head, target_num_feat, source_num_feat -> use all att head and CLS as target
                    last_layer_att_score = last_layer_att_score.cpu().numpy().tolist()

                else:
                    last_layer_att_score = []
                    activation_score = []


                if self.task_q or self.task_vqa:
                    score, label = logit.max(1)
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        quesid2ans[qid] = int(l)
                        results.append(
                            {
                                "questionId": qid,
                                "prediction": l.tolist(),
                                "attention": last_layer_att_score,
                            }
                        )



        # exp_name = args.output.split('/')[-1]
        # evaluator.save_json(results, 'snap/agqa/{output}/val_attentions_cross_2.json'.format(output=exp_name))

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)

        if self.task_q or self.task_vqa:
            return quesid2ans
        else:
            return quesid2ans, hg_quesid2ans






    def test(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        results = []


        for i, datum_tuple in enumerate(loader):
            ques_id, vid_id, feats, boxes, sent, target = datum_tuple[:6]  # todo: changed to 10

            features = convert_sents_to_features(
                sent, self.max_seq_length, self.tokenizer)

            bsz = features.size(0)

            # process question features
            input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
            input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
            segment_ids = torch.tensor(np.array([f.segment_ids for f in features]), dtype=torch.long)

            attention = []

            with torch.no_grad():
                if torch.cuda.is_available():
                    feats, boxes = feats.cuda(), boxes.cuda()
                    input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()



                logit, attn_probs = self.model(feats, boxes, input_ids=input_ids, input_masks=input_mask,
                                               segment_ids=segment_ids, rel_segment_ids=None,
                                               rel_tgt_mask=None, act_segment_ids=None,
                                               act_tgt_mask=None, hg_mask=None,
                                               rel_tgt_ids=None, act_tgt_ids=None)

                if args.output_attention:
                    last_layer_att_score = torch.squeeze(attn_probs[attn_idx[args.cross_attn_type]][-1]['attn'][:, :, 0,
                                                         :])  # batch_size, att_head, target_num_feat, source_num_feat -> use all att head and CLS as target
                    # print(last_layer_att_score.shape)
                    last_layer_att_score = last_layer_att_score.cpu().numpy().tolist()

                    # activation_score = torch.squeeze(attn_probs[-2][-1])
                    # activation_score = activation_score.cpu().numpy().tolist()

                else:
                    last_layer_att_score = []
                    activation_score = []

                score, label = logit.max(1)

                for qid, l in zip(ques_id, label.cpu().numpy()):
                    # ans = dset.label2ans[l]
                    quesid2ans[qid] = int(l)
                    results.append(
                        {
                            "questionId": qid,
                            "prediction": l.tolist(),
                            "attention": last_layer_att_score
                        }
                    )



        exp_name = args.output.split('/')[-1]
        evaluator.save_json(results, 'snap/agqa/{output}/val_attentions_cross_2.json'.format(output=exp_name))



        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)

        return quesid2ans






    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluateOverall(quesid2ans)



    def evaluateAllQtypes(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluateAllQtypes(quesid2ans)



    def evaluateTestSplits(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)


        if args.indirect_ref:
            return evaluator.evaluateAllQtypes(quesid2ans), evaluator.evaluateIndirectRef(quesid2ans)


        elif args.novel_comp:
            return evaluator.evaluateNovelComp(quesid2ans)


        elif args.comp_steps:
            return evaluator.evaluateCompSteps(quesid2ans)




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
    # torch.cuda.empty_cache()

    # Build Class
    agqa = AGQA()


    # Load Model
    if args.load is not None:
        agqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False  # Always loading all data in test


        if 'valid' in args.test:
            result = agqa.evaluateAllQtypes(
                agqa.valid_tuple, dump=None)


            log_str = "Valid:\n Overall %0.2f\n" % (result[0] * 100.) + \
                      "object-relationship: %0.2f\n" % (result[1] * 100.) + \
                      "relationship-action: %0.2f\n" % (result[2] * 100.) + \
                      "object-action: %0.2f\n" % (result[3] * 100.) + \
                      "superlative: %0.2f\n" % (result[4] * 100.) + \
                      "sequencing: %0.2f\n" % (result[5] * 100.) + \
                      "exists: %0.2f\n" % (result[6] * 100.) + \
                      "duration-comparison: %0.2f\n" % (result[7] * 100.) + \
                      "action-recognition: %0.2f\n" % (result[8] * 100.)

            print(log_str, flush=True)



        if 'test' in args.test:
            testData = get_tuple('test', bs=args.batch_size,
                                 shuffle=False, drop_last=False)

            if args.indirect_ref:
                result, result2 = agqa.evaluateTestSplits(testData,
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
                           "localization recall open: %0.2f\n" % (result2[8] * 100.) + \
                           "\nPRECISION\n" + \
                           "object precision: %0.2f\n" % (result2[9] * 100.) + \
                           "object precision binary: %0.2f\n" % (result2[10] * 100.) + \
                           "object precision open: %0.2f\n" % (result2[11] * 100.) + \
                           "action precision: %0.2f\n" % (result2[12] * 100.) + \
                           "action precision binary: %0.2f\n" % (result2[13] * 100.) + \
                           "action precision open: %0.2f\n" % (result2[14] * 100.) + \
                           "localization precision: %0.2f\n" % (result2[15] * 100.) + \
                           "localization precision binary: %0.2f\n" % (result2[16] * 100.) + \
                           "localization precision open: %0.2f\n" % (result2[17] * 100.)

                print(log_str, flush=True)
                print(log_str2, flush=True)




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

                log_str = "\nTest Compositional Steps:\n" \
                          "overall: %0.2f\n" % (result[0] * 100.) + \
                          "overall binary: %0.2f\n" % (result[1] * 100.) + \
                          "overall open: %0.2f\n" % (result[2] * 100.)

                print(log_str, flush=True)





            else:
                result = agqa.evaluateAllQtypes(testData, dump=os.path.join(args.output, 'test_data.json'))

                log_str = "\nTest: Overall %0.2f\n" % (result[0] * 100.) + \
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
        # print("Train Oracle: %0.2f" % (agqa.oracle_score(agqa.train_tuple) * 100), flush=True)
        # print('Splits in Train data:', agqa.train_tuple.dataset.splits, flush=True)
        # if agqa.valid_tuple is not None:
        #     print('Splits in Valid data:', agqa.valid_tuple.dataset.splits)
        #     print("Valid Oracle: %0.2f" % (agqa.oracle_score(agqa.valid_tuple) * 100), flush=True)
        # else:
        #     print("DO NOT USE VALIDATION")

        agqa.train(agqa.train_tuple, agqa.valid_tuple)




