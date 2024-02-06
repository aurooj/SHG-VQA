# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

from cv2 import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import src.utils
from src.param import args
from src.utils import load_obj_tsv, load_spatial_gqa
from src.data_transforms import DataTransforms, QAInputArrange
from src.visualization_tools.vis_utils import trim_keyframes, get_vocab, get_act_cls, trim_keyframes_test, \
    get_vocab_dict, create_relationship_data, uniform_subsample, get_action_dictionaries, get_merged_data

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000
CLIP_LEN = args.CLIP_LEN
eval_star_humans = False

class STARDataset:
    """
    A AGQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """

    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        # path configs

        # plase replace following data path to your local path
        # all data can be download from our homepages (http://star.csail.mit.edu or https://bobbywu.com/STAR)
        root_dir = '../../'
        annotation_dir = root_dir + 'STAR/data/classes/'

        self.num_rel = args.num_rel
        self.num_situations = args.num_situations
        self.num_act = args.num_act
        self.action_classes = get_act_cls(annotation_dir)

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            if split == 'test':
                if eval_star_humans:
                    self.data.extend(json.load(open(root_dir + f'STAR/data/STAR_Human_V0.5/{args.qtype}_%s.json' % split)))
                else:
                    self.data.extend(json.load(open(root_dir + 'STAR/data/STAR_%s.json' % split)))
            else:
                self.data.extend(json.load(open(root_dir + 'STAR/data/STAR_%s_updated.json' % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        rel_triplet_path = os.path.join(annotation_dir, 'relationship_triplets.json')
        if os.path.isfile(rel_triplet_path):
            self.rel_triplets_data = src.utils.load_pickle(rel_triplet_path)
        else:
            print(f"File not found: {rel_triplet_path}, creating one...")
            total_data = [json.load(open(root_dir + 'STAR/data/STAR_%s_updated.json' % split)) for split in ['val', 'train']]
            total_data = self.merge_lists(total_data)
            self.rel_triplets_data = create_relationship_data(total_data, annotation_dir)

        act_path = os.path.join(annotation_dir, 'action_dictionaries.json')
        if os.path.isfile(act_path):
            self.action_data = src.utils.load_pickle(act_path)
        else:
            print(f"File not found: {act_path}, creating one...")
            self.action_data = get_action_dictionaries(annotation_dir)


        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        self.ans2label = {'0': 0, '1': 1, '2': 2,
                          '3': 3}  # dummy dictionary for AGQA multiple choices (we have 4 choices to pick the answer from)

    def merge_lists(self, lists):
        total = []
        for td in lists:
            total.extend(td)
        return total

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)



"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""


class STARTorchDataset(Dataset):
    def __init__(self, dataset: STARDataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        root_dir = '../../'
        self.annotation_dir = root_dir + 'STAR/STAR_Benchmark/annotations/STAR_classes/'
        self.video_dir = root_dir + 'ActionGenome/ActionGenome/dataset/ag/videos/'
        self.frame_dir = root_dir + 'ActionGenome/ActionGenome/dataset/ag/frames/'
        self.pose_dir = root_dir + 'STAR/data/pose/'
        self.qtype = args.qtype
        print("Training/evaluating for question type:" + self.qtype)
        self.fps = pickle.load(open(self.annotation_dir + 'video_fps', 'rb'))
        self.clip_len = args.CLIP_LEN
        self.num_rel = args.num_rel
        self.num_situations = args.num_situations
        self.num_act = args.num_act
        self.vid_ids_to_augment = src.utils.load_json(root_dir+ 'STAR/data/nopred_nofeas_vid_ids_train.json')

        if self.raw_dataset.name == 'test':
            assert (args.augment_type == 'no_aug' or args.augment_type == 'no_aug_slowfast') and self.raw_dataset.name == 'test'
        is_eval = self.raw_dataset.name in ['test', 'valid']

        self.transform = DataTransforms(args.augment_type)
        self.qa_arrange = QAInputArrange(args.qa_arrange_type)

        # load relation and object vocabulary
        self.obj_vocab, self.rel_vocab = get_vocab(self.annotation_dir)
        self.action_classes = get_act_cls(self.annotation_dir)

        # Only kept the data with loaded image features
        self.data = []
        if args.merge_data:
            print('Using merged data for training the model...')
            if is_eval or args.merge_all:
                for datum in self.raw_dataset.data:
                        self.data.append(datum)

            else:
                self.merged_data = get_merged_data(self.raw_dataset.data)
                for qtype, data in self.merged_data.items():
                    for datum in data:
                        self.data.append(datum)

        else:
            for datum in self.raw_dataset.data:
                if self.qtype in datum['question_id']:
                    self.data.append(datum)
                elif self.qtype in ['Prediction', 'Feasibility'] and datum['video_id'] in self.vid_ids_to_augment and not is_eval:
                    self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        vid_id = datum['video_id']
        ques_id = datum['question_id']

        question = datum['question']
        choices = {ch['choice_id']: ch['choice'] for ch in datum['choices']}
        qa_out = self.qa_arrange.qa_prep(question, choices)

        assert (isinstance(qa_out, str) and args.qa_arrange_type in ['add_sep_all', 'no_sep_all']) or \
               (isinstance(qa_out, dict) and args.qa_arrange_type in ['add_sep', 'no_sep'])

        if isinstance(qa_out, dict):
            qa0 = qa_out['qa0']
            qa1 = qa_out['qa1']
            qa2 = qa_out['qa2']
            qa3 = qa_out['qa3']
        else:
            qa0, qa1, qa2, qa3 = 0, 0, 0, 0

        # get key frames
        if self.raw_dataset.name == 'test':
            frame_list = os.listdir(os.path.join(self.frame_dir, f"{datum['video_id']}.mp4"))
            frame_ids = [fn.split('.')[0] for fn in frame_list]
            trimmed_frame_ids = trim_keyframes_test(datum, frame_ids, self.fps, max_show_num=CLIP_LEN)
        else:
            trimmed_frame_ids = trim_keyframes(datum, self.fps,
                                               max_show_num=CLIP_LEN)  # max_show_num=0 to return all frames

        select = []
        for i in range(len(trimmed_frame_ids)):
            frame = cv2.imread(self.frame_dir + '/' + f'{vid_id}.mp4' + '/' + trimmed_frame_ids[i] + '.png')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            select.append(frame)

        frames = torch.tensor(np.array(select))
        frames = self.transform.transform(frames)
        if args.backbone in ["slowfast_r50","slowfast_r101"]:
            mask_shape = 8 * 8 * 8 + 1
        else:
            mask_shape = 8 * 7 * 7 + 1  # t=8 (after R3D+conv), h=w=7, +1 for cls token
        boxes = np.ones(mask_shape)  # dummy value for now

        # Create target
        if 'answer' in datum:
            situations = {k: v for k, v in datum['situations'].items() if k in trimmed_frame_ids}

            rel_trplts_tokens, lengths, unpad_rels = self.tokenize_relation_triplets(situations,
                                                                                     self.raw_dataset.rel_triplets_data[
                                                                                         'rel_triplets_rp2idx'])
            rel_trplts_tokens, indices = uniform_subsample(torch.tensor(rel_trplts_tokens), CLIP_LEN, 0)
            lengths = [lengths[idx] for idx in indices]

            # get actions
            action_tokens, act_lengths, unpad_acts = self.tokenize_actions(situations, self.raw_dataset.action_data[
                'actions_rp2idx'])
            action_tokens, indices = uniform_subsample(torch.tensor(action_tokens), CLIP_LEN, 0)
            act_lengths = [act_lengths[idx] for idx in indices]

            act_mask = action_tokens.clone()
            act_mask[act_mask > 0] = 1
            rel_mask = rel_trplts_tokens.clone()
            rel_mask[rel_mask > 0] = 1
            hg_mask = torch.cat([act_mask, rel_mask], dim=1)

            # trim lengths to max_allowed lengths
            a_lens = np.array(act_lengths)

            a_lens[a_lens > self.num_act] = self.num_act
            r_lens = np.array(lengths)
            r_lens[r_lens > self.num_rel] = self.num_rel

            label = datum['answer_choice']
            target = torch.zeros(self.raw_dataset.num_answers)
            target[label] = 1.
  
            return ques_id, vid_id, frames, boxes, qa_out, rel_trplts_tokens, r_lens, action_tokens, hg_mask, \
                   qa0, qa1, qa2, qa3, a_lens, target # we will add choices later
        else:
            hg_mask = np.ones((CLIP_LEN, self.num_act+self.num_rel), dtype=int)
            rel_trplts_tokens, r_lens, action_tokens, a_lens = [], [], [], []
            return ques_id, vid_id, frames, boxes, qa_out, rel_trplts_tokens, r_lens, action_tokens, hg_mask,\
                   qa0, qa1, qa2, qa3, a_lens  # we will add choices later


    def tokenize_actions(self, situations, act_dict):
        actions = [[act for act in situation['actions']] for k, situation in situations.items()]
        # tokenize each action based on its label
        action_tokens = [[act_dict[k] for k in rp] for rp in actions]
        actions_out, lengths, unpadded_act = self.pad_actions(action_tokens, max_act_allowed=self.num_act)
        return actions_out, lengths, unpadded_act

    def pad_actions(self, actions, max_act_allowed=3):
        lengths = [len(act) for act in actions]
        pad_token = 0  # action class labels start from 1
        actions_out = [(act + [pad_token] * (max_act_allowed - len(act)))[:max_act_allowed] for
                                   act in actions]
        return actions_out, lengths, actions

    def tokenize_relation_triplets(self, situations, rel_triplets_rp2idx):
        #create relation triplets from rel_pairs and rel_labels
        rel_trplts = [[(rp[0], rl, rp[1]) for rp, rl in zip(situation['rel_pairs'], situation['rel_labels'])] for
                      k, situation in situations.items()]
        #tokenize each triplet based on its label
        rel_trplts_tokens = [[rel_triplets_rp2idx[k] for k in rp] for rp in rel_trplts]

        return self.pad_relation_triplets(rel_trplts_tokens, max_rel_allowed=self.num_rel)

    def pad_relation_triplets(self, rel_triplets_tokens, max_rel_allowed=8):
        #not using max len for padding currently
        lengths = [len(rel_trp) for rel_trp in rel_triplets_tokens]
        pad_token = 0 #relation class labels start from 1
        rel_triplets_tokens_out = [(rel_trp + [pad_token]*(max_rel_allowed-len(rel_trp)))[:max_rel_allowed] for rel_trp in rel_triplets_tokens]
        return rel_triplets_tokens_out, lengths,   rel_triplets_tokens


class STAREvaluator:
    def __init__(self, dataset: STARDataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['answer_choice']
            if ans == label:
                score += int(ans == label)
        return score / len(quesid2ans)

    def save_json(self, data, file_path):
        with open(file_path, "wb") as f:
            json.dump(data, f)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'wb') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)
