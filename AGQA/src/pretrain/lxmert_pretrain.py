# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os
import random

import gc
import sys
from datetime import datetime

# from GPUtil import showUtilization as gpuUsage

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.param import args
from src.pretrain.lxmert_data import InputExample, LXMERTDataset, LXMERTTorchDataset, LXMERTPatchesTorchDataset, LXMERTEvaluator
from src.lxrt.entry import set_visual_config
from src.lxrt.tokenization import BertTokenizer
from src.lxrt.modeling_capsbert import LXRTPretraining

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(log_dir=os.path.join('runs', args.output, datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))

DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')

print(args.task_mask_lm, args.task_obj_predict, args.task_matched, args.task_qa, args.task_contrastive, file=sys.stderr)
def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1, patches=False) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))

    # Build dataset, data loader, and evaluator.
    dset = LXMERTDataset(splits, qa_sets=qa_sets)
    if patches:
        tset = LXMERTPatchesTorchDataset(dset, topk)
    else:
        tset = LXMERTTorchDataset(dset, topk)

    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True
    )
    evaluator = LXMERTEvaluator(dset)
    print(file=sys.stderr)

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)


train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True, patches=args.patches)
valid_batch_size = args.batch_size #2048 if args.multiGPU else args.batch_size
print("valid batch size:{}".format(valid_batch_size), file=sys.stderr)
valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False, topk=5000, patches=args.patches)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids,
                 visual_feats, obj_labels,
                 is_matched, ans):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = args.word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_feat(feats):
    mask_feats = feats.copy()
    # print(mask_feats.shape)
    feat_mask = np.zeros(len(feats), dtype=np.float32)
    # for i in range(len(feats)):
    #     prob = random.random()
    #     # mask token with probability
    #     if prob < args.obj_mask_rate:
    #         prob /= args.obj_mask_rate
    #
    #         # 80% randomly change token to zero feat
    #         if prob < 0.8:
    #             mask_feats[i, :] = 0.
    #
    #         # 10% randomly change token to random feat
    #         elif prob < 0.9:
    #             mask_feats[i, :] = train_tuple.torchdset.random_feat()
    #         # -> rest 10% randomly keep current feat
    #
    #         # Need to predict this feat
    #         feat_mask[i] = 1.

    return mask_feats, feat_mask


def convert_example_to_features(example: InputExample, max_seq_length, tokenizer)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens = tokenizer.tokenize(example.sent.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    masked_tokens, masked_label = random_word(tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    lm_label_ids = ([-1] + masked_label + [-1])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    feat, boxes = example.visual_feats
    obj_labels, obj_confs = example.obj_labels
    attr_labels, attr_confs = example.attr_labels

    # Mask Image Features:
    masked_feat, feat_mask = random_feat(feat)

    # QA answer label
    if example.label is None or len(example.label) == 0 or example.is_matched != 1:
        # 1. No label 2. Label is pruned 3. unmatched visual + language pair
        ans = -1
    else:
        keys, values = zip(*example.label.items())
        if len(keys) == 1:
            ans = keys[0]
        else:
            value_sum = sum(values)
            prob = [value / value_sum for value in values]
            choice = np.random.multinomial(1, prob).argmax()
            ans = keys[choice]

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        visual_feats=(feat, boxes),
        obj_labels={
            'obj': (obj_labels, obj_confs),
            'attr': (attr_labels, attr_confs),
            'feat': (feat, feat_mask),
        },
        is_matched=example.is_matched,
        ans=ans,
    )
    return features


LOSSES_NAME = ('Mask_LM', 'Matched', 'QA', 'CTR', 'Obj', 'Attr', 'Feat')


class LXMERT:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        print(args.task_mask_lm, args.task_obj_predict, args.task_matched, args.task_qa, args.task_contrastive,
              file=sys.stderr)

        # Build model
        set_visual_config(args)
        self.model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            task_contrastive=args.task_contrastive,
            visual_losses=args.visual_losses,
            num_answers=train_tuple.dataset.answer_table.num_answers,
            skip_connection=args.skip_connection,
            shared_weights=args.shared_weights,
            cross_attn = args.cross_attn,
            cross_attn_type = args.cross_attn_type,
            freeze_weights = args.freeze_weights,
            patches=args.patches,
            vit_init=args.vit_init,
            start_index=args.start_index,
            no_caps = args.no_caps
        )


        # Weight initialization and loading
        if args.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.", file=sys.stderr)
            self.model.apply(self.model.init_bert_weights)
        if args.load is not None:
            self.load(args.load)
        if args.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(args.load_lxmert)

        # GPU Options
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if args.multiGPU:
                self.model = nn.DataParallel(self.model)

        print(self.model, file=sys.stderr)

    def forward(self, examples):
        train_features = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        # language Inputs

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).to(self.device)

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).to(self.device)
        h=7 #todo: make it configurable
        pos = torch.from_numpy(np.stack([torch.ones(h * h+1) for f in train_features])).to(self.device)

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).to(self.device)

        # Visual Prediction
        obj_labels = {}
        for key in ('obj', 'attr', 'feat'):
            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).to(self.device)
            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).to(self.device)
            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
            obj_labels[key] = (visn_labels, visn_mask)

        # Joint Prediction
        matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long).to(self.device)
        ans = torch.from_numpy(np.stack([f.ans for f in train_features])).to(self.device)

        """
        
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """
        loss, losses, ans_logit = self.model(
            input_ids, segment_ids, input_mask, lm_labels,
            feats, pos, obj_labels, matched_labels, ans
        )
        return loss, losses.detach().cpu(), ans_logit

    def train_batch(self, optim, batch):
        optim.zero_grad()
        loss, losses, ans_logit = self.forward(batch)
        if args.multiGPU:
            loss = loss.mean()
            losses = losses.mean(0)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        optim.step()

        return loss.item(), losses.cpu().numpy(), ans_logit

    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses, ans_logit = self.forward(batch)
            if args.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy(), ans_logit

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        train_ld = train_tuple.loader

        # Optimizer
        from src.lxrt.optimization import BertAdam
        batch_per_epoch = len(train_ld)
        t_total = int(batch_per_epoch * args.epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        print("Batch per epoch: %d" % batch_per_epoch, file=sys.stderr)
        print("Total Iters: %d" % t_total, file=sys.stderr)
        print("Warm up Iters: %d" % warmup_iters, file=sys.stderr)
        optim = BertAdam(self.model.parameters(), lr=args.lr, warmup=warmup_ratio, t_total=t_total)

        # Train
        best_eval_loss = 9595.
        for epoch in range(args.epochs):
            # Train
            self.model.train()
            total_loss = 0.
            total_losses = 0.
            uid2ans = {}
            for batch in tqdm(train_ld, total=len(train_ld)):
                loss, losses, logit = self.train_batch(optim, batch)
                total_loss += loss
                total_losses += losses

                if args.task_qa:
                    score, label = logit.max(1)
                    for datum, l in zip(batch, label.cpu().numpy()):
                        uid = datum.uid
                        ans = train_tuple.dataset.answer_table.id2ans(l)
                        uid2ans[uid] = ans

                # del logit, loss, losses
                # gc.collect()

            # gpuUsage()
            print("The training loss for Epoch %d is %0.4f" % (epoch, total_loss / batch_per_epoch), file=sys.stderr)
            losses_str = "The losses are "
            print(total_losses, file=sys.stderr)

            if isinstance(total_losses, np.ndarray):
                total_losses = total_losses.tolist()
            if isinstance(total_losses[0], list):
                total_losses = total_losses[0]

            for name, loss in zip(LOSSES_NAME, total_losses):
                print(name, loss, file=sys.stderr)
                losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)
                writer.add_scalar('Loss/train_%s'%name, loss, epoch)

            print(losses_str, file=sys.stderr)

            if args.task_qa:
                acc, dset2acc = train_tuple.evaluator.evaluate(uid2ans, pprint=True)
                # logging in tensorboard
                sorted_keys = sorted(dset2acc.keys())
                for key in sorted_keys:
                    writer.add_scalar('Accuracy/train_%s' % key, dset2acc[key], epoch)

            #to handle GPU OOM error
            # torch.cuda.empty_cache()
            # Eval
            avg_eval_loss = self.evaluate_epoch(eval_tuple, iters=-1, epoch=epoch)

            # Save
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                writer.add_scalar('Loss/eval_loss', avg_eval_loss, epoch)
                self.save("BEST_EVAL_LOSS")
            self.save("Epoch%02d" % (epoch+1))

    def evaluate_epoch(self, eval_tuple: DataTuple, iters: int=-1, epoch: int=-1):
        self.model.eval()
        eval_ld = eval_tuple.loader
        total_loss = 0.
        total_losses = 0.
        uid2ans = {}
        print("inside evaluation code..", file=sys.stderr)
        # gpuUsage()
        for i, batch in enumerate(eval_ld):
            loss, losses, logit = self.valid_batch(batch)
            total_loss += loss
            total_losses += losses
            if args.task_qa:
                score, label = logit.max(1)
                for datum, l in zip(batch, label.cpu().numpy()):
                    uid = datum.uid
                    ans = train_tuple.dataset.answer_table.id2ans(l)
                    uid2ans[uid] = ans

            # del logit, loss, losses
            # gc.collect()

            if i == iters:
                break

        print("The valid loss is %0.4f" % (total_loss / len(eval_ld)), file=sys.stderr)


        if isinstance(total_losses, np.ndarray):
            total_losses = total_losses / len(eval_ld)
            total_losses = total_losses.tolist()
        if isinstance(total_losses[0], list):
            total_losses = total_losses[0]

        losses_str = "The losses are "
        for name, loss in zip(LOSSES_NAME, total_losses):
            losses_str += "%s: %0.4f " % (name, loss)
            writer.add_scalar('Loss/valid_%s'%name, loss, epoch)

        print(losses_str, file=sys.stderr)

        if args.task_qa:
            acc, dset2acc = eval_tuple.evaluator.evaluate(uid2ans, pprint=True)
            #logging in tensorboard
            sorted_keys = sorted(dset2acc.keys())
            for key in sorted_keys:
                writer.add_scalar('Accuracy/valid_%s'%key, dset2acc[key], epoch)

        return total_loss / len(eval_ld)

    def save(self, name):
        if os.path.exists(args.output):
            torch.save(self.model.state_dict(),
                   os.path.join(args.output, "%s_LXRT.pth" % name))
        else:
            os.makedirs(args.output)
            torch.save(self.model.state_dict(),
                       os.path.join(args.output, "%s_LXRT.pth" % name))

    def load(self, path):
        print("Load BERT extractor from %s" % path, file=sys.stderr)
        state_dict = torch.load("%s_LXRT.pth" % path)
        self.model.load_state_dict(state_dict)

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path, file=sys.stderr)
        state_dict = torch.load("%s_LXRT.pth" % path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print(file=sys.stderr)
        print("Keys in loaded but not in model:", file=sys.stderr)
        for key in sorted(load_keys.difference(model_keys)):
            print(key, file=sys.stderr)
        print(file=sys.stderr)
        print("Keys in model but not in loaded:", file=sys.stderr)
        for key in sorted(model_keys.difference(load_keys)):
            print(key, file=sys.stderr)
        print(file=sys.stderr)

        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    print("print training objectives...", file=sys.stderr)
    print("MLM={mlm}, Obj={obj}, ITM={itm}, QA={qa}, Contrastive={ctr}".format(mlm=args.task_mask_lm,
                                                                               obj=args.task_obj_predict,
                                                                               itm=args.task_matched,
                                                                               qa=args.task_qa,
                                                                               ctr=args.task_contrastive),
          file=sys.stderr)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    lxmert = LXMERT(max_seq_length=20)


    lxmert.train(train_tuple, valid_tuple)
    writer.close()
