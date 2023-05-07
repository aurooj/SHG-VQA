# coding=utf-8
# Copyleft 2019 project LXRT.
import time
import ffmpeg
import h5py

import json
import os
import pickle
import copy

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import src.utils
from src.param import args
from src.utils import load_obj_tsv, load_spatial_gqa
from src.data_transforms import DataTransforms, QAInputArrange
from src.visualization_tools.vis_utils import trim_keyframes, get_vocab, get_act_cls, \
    get_vocab_dict, create_relationship_data, uniform_subsample, get_action_dictionaries, get_scene_graphs, get_answer_vocab


CLIP_LEN = 16


class AGQADataset:
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        # path configs

        # todo: Plase replace following data path to your local path
        root_dir = '//'
        annotation_dir = root_dir + '/'

        self.num_rel = 8
        self.num_situations = 16
        self.num_act = 3
        self.action_classes = get_act_cls(annotation_dir)



        # Loading datasets to data
        self.data = []


        for split in self.splits:
            if split == 'test':
                with open(root_dir + 'data/test_balanced.json', 'rb') as f:
                    self.data = pickle.load(f)
                    f.close()


                if args.novel_comp:
                    newData = [q for q in self.data if q["novel_comp"] == 1]
                    self.data = newData

                if args.comp_steps:
                    newData = [q for q in self.data if q["more_steps"] == 1]
                    self.data = newData


            else:
                if split == 'train':
                    if not args.train_sub_set:
                        with open(root_dir + 'data/train_balanced.json', 'rb') as f:
                            self.data = pickle.load(f)
                            f.close()

                    if args.train_sub_set:
                        with open(root_dir + 'data/new-sub-train.json', 'rb') as f:
                            self.data = pickle.load(f)
                            f.close()


                    if args.novel_comp:
                        newData = [q for q in self.data if q["novel_comp"] == 0]
                        self.data = newData

                    if args.comp_steps:
                        newData = [q for q in self.data if q["more_steps"] == 0]
                        self.data = newData



                elif split == 'valid':
                    with open(root_dir + 'data/valid_balanced.json', 'rb') as f:
                        self.data = pickle.load(f)
                        f.close()

                    if args.novel_comp:
                        newData = [q for q in self.data if q["novel_comp"] == 1]
                        self.data = newData

                    if args.comp_steps:
                        newData = [q for q in self.data if q["more_steps"] == 1]
                        self.data = newData


            # load answer vocab
            self.answerVocab = get_answer_vocab(annotation_dir + '/trainVal_vocab.json')




        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        rel_triplet_path = os.path.join(annotation_dir, 'relationship_triplets.json')
        if os.path.isfile(rel_triplet_path):
            self.rel_triplets_data = src.utils.load_pickle(rel_triplet_path)
        else:
            print(f"File not found: {rel_triplet_path}, creating one...")


        # load action dictionary
        act_path = os.path.join(annotation_dir, 'action_dictionaries.json')
        if os.path.isfile(act_path):
            self.action_data = src.utils.load_pickle(act_path)
        else:
            print(f"File not found: {act_path}, creating one...")
            self.action_data = get_action_dictionaries(annotation_dir)


        # load relationships for each frame
        with open(annotation_dir + '/frameTriplets.json', 'rb') as f:
            self.frameTriplets = pickle.load(f)
            f.close()

        # load actions for each frame
        with open(annotation_dir + '/frameActions.json', 'rb') as f:
            self.frameActions = pickle.load(f)
            f.close()


        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }



    def merge_lists(self, lists):
        total = []
        for td in lists:
            total.extend(td)
        return total

    @property
    def num_answers(self):
        return len(self.answerVocab)
        # return 171

    def __len__(self):
        return len(self.data)



class AGQATorchDataset(Dataset):
    def __init__(self, dataset: AGQADataset):
        super().__init__()
        self.raw_dataset = dataset
        self.data = self.raw_dataset.data

        #todo: REPLACE WITH YOUR DIRECTORIES
        root_dir = '/'
        self.annotation_dir = root_dir + '/'
        self.video_dir = '/datasets/Charades/data/Charades_v1_480'
        self.frame_dir = '/datasets/ActionGenome/dataset/ag/frames'

        self.clip_len = 16
        self.num_rel = 8
        self.num_situations = 16
        self.num_act = 3



        # clip len == 16
        with open(self.annotation_dir + '/trimmed_frame_ids.json', 'rb') as f:
            self.frame_ids = pickle.load(f)
            f.close()



        if self.raw_dataset.name == 'test':
            assert args.augment_type == 'no_aug' and self.raw_dataset.name == 'test'


        self.transform = DataTransforms(args.augment_type)

        # load actions
        self.action_classes = get_act_cls(self.annotation_dir)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        vid_id = datum['video_id']
        ques_id = datum['question_id']
        question = datum['question']

        if not args.task_q:
            cv2.setNumThreads(1)
            trimmed_frame_ids = self.frame_ids[vid_id]
            select = []
            for i in range(len(trimmed_frame_ids)):
                frame = cv2.imread(self.frame_dir + '/' + f'{vid_id}.mp4' + '/' + trimmed_frame_ids[i] + '.png')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                select.append(frame)
            frames = torch.as_tensor(np.array(select))
            frames = self.transform.transform(frames)

        mask_shape = 8 * 7 * 7 + 1  # t=8 (after R3D+conv), h=w=7, +1 for cls token
        boxes = np.ones(mask_shape)  # dummy value for now



        # Create target
        if args.task_hgqa or args.task_vhga:
            if args.test != None:
                hg_mask = np.ones((CLIP_LEN, self.num_act + self.num_rel), dtype=int)
                rel_trplts_tokens, r_lens, action_tokens, a_lens = [], [], [], []
                return ques_id, vid_id, frames, boxes, question, rel_trplts_tokens, r_lens, action_tokens, hg_mask, \
                       a_lens

            else:
                vid = datum["video_id"]
                trimmed_frame_ids = self.frame_ids[vid_id]

                rel_trplts_tokens, lengths, unpad_rels = self.tokenize_relation_triplets(vid, trimmed_frame_ids,
                                                                                         self.raw_dataset.frameTriplets,
                                                                                         self.raw_dataset.rel_triplets_data[
                                                                                             'rel_triplets_rp2idx'])
                rel_trplts_tokens, indices = uniform_subsample(torch.as_tensor(rel_trplts_tokens), CLIP_LEN, 0)
                lengths = [lengths[idx] for idx in indices]

                # get actions
                action_tokens, act_lengths, unpad_acts = self.tokenize_actions(vid, trimmed_frame_ids,
                                                                               self.raw_dataset.frameActions,
                                                                               self.raw_dataset.action_data[
                                                                                   'actions_rp2idx'])
                action_tokens, indices = uniform_subsample(torch.as_tensor(action_tokens), CLIP_LEN, 0)
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

                target = torch.zeros(self.raw_dataset.num_answers)
                label = datum["answer"]
                indx = self.raw_dataset.answerVocab[label]
                target[indx] = 1.

                return ques_id, vid_id, frames, boxes, question, rel_trplts_tokens, r_lens, action_tokens, hg_mask, \
                       a_lens, target



        if args.task_q:
            if args.test != None:
                target = torch.zeros(self.raw_dataset.num_answers)

            else:
                target = torch.zeros(self.raw_dataset.num_answers)
                label = datum["answer"]
                indx = self.raw_dataset.answerVocab[label]
                target[indx] = 1.

            return ques_id, vid_id, boxes, question, target




        elif args.task_vqa:
            if args.test != None:
                target = torch.zeros(self.raw_dataset.num_answers)
            else:
                target = torch.zeros(self.raw_dataset.num_answers)
                label = datum["answer"]
                indx = self.raw_dataset.answerVocab[label]
                target[indx] = 1.

            return ques_id, vid_id, frames, boxes, question, target





    def tokenize_actions(self, vid, trimmedFrames, acts, act_dict):
        actions = [acts[vid][x] for x in trimmedFrames]

        # tokenize each action based on its label
        action_tokens = [[act_dict[k] for k in rp] for rp in actions]

        actions_out, lengths, unpadded_act = self.pad_actions(action_tokens)
        return actions_out, lengths, unpadded_act



    def pad_actions(self, actions, max_act_allowed=3):
        lengths = [len(act) for act in actions]
        pad_token = 0  # action class labels start from 1
        actions_out = [(act + [pad_token] * (max_act_allowed - len(act)))[:max_act_allowed] for
                       act in actions]

        return actions_out, lengths, actions




    def tokenize_relation_triplets(self, vid, trimmedFrames, triplets, rel_triplets_rp2idx):
        rel_trplts = [triplets[vid][x] for x in trimmedFrames]
        rel_trplts_tokens = [[rel_triplets_rp2idx[k] for k in rp] for rp in rel_trplts]
        return self.pad_relation_triplets(rel_trplts_tokens)



    def pad_relation_triplets(self, rel_triplets_tokens, max_rel_allowed=8):
        # not using max len for padding currently
        lengths = [len(rel_trp) for rel_trp in rel_triplets_tokens]
        pad_token = 0  # relation class labels start from 1
        rel_triplets_tokens_out = [(rel_trp + [pad_token] * (max_rel_allowed - len(rel_trp)))[:max_rel_allowed] for
                                   rel_trp in rel_triplets_tokens]
        return rel_triplets_tokens_out, lengths, rel_triplets_tokens




class AGQAEvaluator:
    def __init__(self, dataset: AGQADataset):
        self.dataset = dataset
        self.answerVocab = dataset.answerVocab

        self.index_to_ans = list(self.answerVocab.keys())



    def evaluateOverall(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['answer']

            ansIndx = int(self.answerVocab[label])
            if ans == ansIndx:
                score += int(ans == ansIndx)
        return score / len(quesid2ans)




    def evaluateAllQtypes(self, quesid2ans: dict):
        score = 0.

        allBinary = 0.
        allBinary_len = 0

        allOpen = 0.
        allOpen_len = 0

        # Semantic Qtypes
        objectSemantic = 0.
        objectSemanticBinary = 0.
        objectSemanticOpen = 0.

        objectSemantic_len = 0
        objectSemanticBinary_len = 0
        objectSemanticOpen_len = 0

        relationshipSemantic = 0.
        relationshipSemantic_len = 0

        actionSemantic = 0.
        actionSemanticBinary = 0.
        actionSemanticOpen = 0.

        actionSemantic_len = 0
        actionSemanticBinary_len = 0
        actionSemanticOpen_len = 0


        # Structure Qtypes
        queryStructure = 0.
        compareStructure = 0.
        chooseStructure = 0.
        logicStructure = 0.
        verifyStructure = 0.

        queryStructure_len = 0
        compareStructure_len = 0
        chooseStructure_len = 0
        logicStructure_len = 0
        verifyStructure_len = 0




        # Reasoning
        object_relationship = 0.
        object_relationshipBinary = 0.
        object_relationshipOpen = 0.

        object_relationship_len = 0
        object_relationshipBinary_len = 0
        object_relationshipOpen_len = 0


        relationship_action = 0.
        relationship_action_len = 0

        object_action = 0.
        object_action_len = 0

        superlative = 0.
        superlativeBinary = 0.
        superlativeOpen = 0.

        superlative_len = 0
        superlativeBinary_len = 0
        superlativeOpen_len = 0


        sequencing = 0.
        sequencingBinary = 0.
        sequencingOpen = 0.

        sequencing_len = 0
        sequencingBinary_len = 0
        sequencingOpen_len = 0


        exists = 0.
        exists_len = 0

        duration_comparison = 0.
        duration_comparisonBinary = 0.
        duration_comparisonOpen = 0.

        duration_comparison_len = 0
        duration_comparisonBinary_len = 0
        duration_comparisonOpen_len = 0


        action_recognition = 0.
        action_recognition_len = 0









        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['answer']     #ground-truth answer
            qType = datum['global']

            ansString = self.index_to_ans[ans]      #convert index to answer

            if datum['ans_type'] == 'binary':
                allBinary_len += 1

            if datum['ans_type'] == 'open':
                allOpen_len += 1


            # Semantics
            if datum['semantic'] == 'object':
                objectSemantic_len += 1
                if datum['ans_type'] == 'binary':
                    objectSemanticBinary_len += 1

                if datum['ans_type'] == 'open':
                    objectSemanticOpen_len += 1


            if datum['semantic'] == 'relation':
                relationshipSemantic_len += 1


            if datum['semantic'] == 'action':
                actionSemantic_len += 1
                if datum['ans_type'] == 'binary':
                    actionSemanticBinary_len += 1

                if datum['ans_type'] == 'open':
                    actionSemanticOpen_len += 1



            # Structure
            if datum['structural'] == 'query':
                queryStructure_len += 1

            if datum['structural'] == 'compare':
                compareStructure_len += 1

            if datum['structural'] == 'choose':
                chooseStructure_len += 1

            if datum['structural'] == 'logic':
                logicStructure_len += 1

            if datum['structural'] == 'verify':
                verifyStructure_len += 1


            # Reasoning
            for q in qType:
                if q == 'obj-rel':
                    object_relationship_len += 1
                    if datum['ans_type'] == 'binary':
                        object_relationshipBinary_len += 1
                    if datum['ans_type'] == 'open':
                        object_relationshipOpen_len += 1

                if q == 'rel-act':
                    relationship_action_len += 1

                if q == 'obj-act':
                    object_action_len += 1

                if q == 'superlative':
                    superlative_len += 1
                    if datum['ans_type'] == 'binary':
                        superlativeBinary_len += 1
                    if datum['ans_type'] == 'open':
                        superlativeOpen_len += 1

                if q == 'sequencing':
                    sequencing_len += 1
                    if datum['ans_type'] == 'binary':
                        sequencingBinary_len += 1
                    if datum['ans_type'] == 'open':
                        sequencingOpen_len += 1

                if q == 'exists':
                    exists_len += 1

                if q == 'duration-comparison':
                    duration_comparison_len += 1
                    if datum['ans_type'] == 'binary':
                        duration_comparisonBinary_len += 1
                    if datum['ans_type'] == 'open':
                        duration_comparisonOpen_len += 1

                if q == 'action-recognition':
                    action_recognition_len += 1





            if ansString == label:
                score += int(ansString == label)
                if datum['ans_type'] == 'binary':
                    allBinary += int(ansString == label)

                if datum['ans_type'] == 'open':
                    allOpen += int(ansString == label)


                # Semantics
                if datum['semantic'] == 'object':
                    objectSemantic += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        objectSemanticBinary += int(ansString == label)

                    if datum['ans_type'] == 'open':
                        objectSemanticOpen += int(ansString == label)


                if datum['semantic'] == 'relation':
                    relationshipSemantic += int(ansString == label)


                if datum['semantic'] == 'action':
                    actionSemantic += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        actionSemanticBinary += int(ansString == label)

                    if datum['ans_type'] == 'open':
                        actionSemanticOpen += int(ansString == label)




                # Structure
                if datum['structural'] == 'query':
                    queryStructure += int(ansString == label)

                if datum['structural'] == 'compare':
                    compareStructure += int(ansString == label)

                if datum['structural'] == 'choose':
                    chooseStructure += int(ansString == label)

                if datum['structural'] == 'logic':
                    logicStructure += int(ansString == label)

                if datum['structural'] == 'verify':
                    verifyStructure += int(ansString == label)




                # Reasoning types
                for q in qType:
                    if q == 'obj-rel':
                        object_relationship += int(ansString == label)
                        if datum['ans_type'] == 'binary':
                            object_relationshipBinary += int(ansString == label)
                        if datum['ans_type'] == 'open':
                            object_relationshipOpen += int(ansString == label)

                    if q == 'rel-act':
                        relationship_action += int(ansString == label)


                    if q == 'obj-act':
                        object_action += int(ansString == label)

                    if q == 'superlative':
                        superlative += int(ansString == label)
                        if datum['ans_type'] == 'binary':
                            superlativeBinary += int(ansString == label)
                        if datum['ans_type'] == 'open':
                            superlativeOpen += int(ansString == label)


                    if q == 'sequencing':
                        sequencing += int(ansString == label)
                        if datum['ans_type'] == 'binary':
                            sequencingBinary += int(ansString == label)
                        if datum['ans_type'] == 'open':
                            sequencingOpen += int(ansString == label)


                    if q == 'exists':
                        exists += int(ansString == label)

                    if q == 'duration-comparison':
                        duration_comparison += int(ansString == label)
                        if datum['ans_type'] == 'binary':
                            duration_comparisonBinary += int(ansString == label)
                        if datum['ans_type'] == 'open':
                            duration_comparisonOpen += int(ansString == label)

                    if q == 'action-recognition':
                        action_recognition += int(ansString == label)


        return [score / len(quesid2ans), allBinary/allBinary_len, allOpen/allOpen_len,
                object_relationship / object_relationship_len, object_relationshipBinary/object_relationshipBinary_len,
                object_relationshipOpen/object_relationshipOpen_len,

                relationship_action / relationship_action_len, object_action / object_action_len,

                superlative / superlative_len, superlativeBinary/superlativeBinary_len, superlativeOpen/superlativeOpen_len,

                sequencing / sequencing_len, sequencingBinary/sequencingBinary_len, sequencingOpen/sequencingOpen_len,
                exists / exists_len,

                duration_comparison / duration_comparison_len, duration_comparisonBinary/duration_comparisonBinary_len,
                duration_comparisonOpen/duration_comparisonOpen_len,

                action_recognition / action_recognition_len,



                objectSemantic/objectSemantic_len, objectSemanticBinary/objectSemanticBinary_len,
                objectSemanticOpen/objectSemanticOpen_len,

                relationshipSemantic/relationshipSemantic_len,

                actionSemantic/actionSemantic_len, actionSemanticBinary/actionSemanticBinary_len,
                actionSemanticOpen/actionSemanticOpen_len,


                queryStructure/queryStructure_len, compareStructure/compareStructure_len, chooseStructure/chooseStructure_len,
                logicStructure/logicStructure_len, verifyStructure/verifyStructure_len

                ]




    def evaluateCompSteps(self, quesid2ans: dict):
        score = 0.
        allBinary = 0.
        allOpen = 0.

        allBinary_len = 0
        allOpen_len = 0

        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]

            label = datum['answer']
            ansString = self.index_to_ans[ans]

            if datum['ans_type'] == 'binary':
                allBinary_len += 1

            if datum['ans_type'] == 'open':
                allOpen_len += 1


            if ansString == label:
                score += int(ansString == label)
                if datum['ans_type'] == 'binary':
                    allBinary += int(ansString == label)

                if datum['ans_type'] == 'open':
                    allOpen += int(ansString == label)


        return [score / len(quesid2ans), allBinary/allBinary_len, allOpen/allOpen_len]




    def evaluateNovelComp(self, quesid2ans: dict):
        overall = 0.
        allBinary = 0.
        allOpen = 0.

        allBinary_len = 0
        allOpen_len = 0


        sequencing = 0.
        sequencingBinary = 0.
        sequencingOpen = 0.

        sequencing_len = 0
        sequencingBinary_len = 0
        sequencingOpen_len = 0


        superlative = 0.
        superlativeBinary = 0.
        superlativeOpen = 0.

        superlative_len = 0
        superlativeBinary_len = 0
        superlativeOpen_len = 0


        duration = 0.
        durationBinary = 0.
        durationOpen = 0.


        duration_len = 0
        durationBinary_len = 0
        durationOpen_len = 0


        object_relationship = 0.
        object_relationshipBinary = 0.
        object_relationshipOpen = 0.


        object_relationship_len = 0
        object_relationshipBinary_len = 0
        object_relationshipOpen_len = 0




        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['answer']     #ground-truth answer

            ansString = self.index_to_ans[ans]      #convert index to answer

            if datum['ans_type'] == 'binary':
                allBinary_len += 1

            if datum['ans_type'] == 'open':
                allOpen_len += 1




            if datum['nc_seq'] == 1:
                sequencing_len += 1
                if datum['ans_type'] == 'binary':
                    sequencingBinary_len += 1
                if datum['ans_type'] == 'open':
                    sequencingOpen_len += 1



            if datum['nc_sup'] == 1:
                superlative_len += 1
                if datum['ans_type'] == 'binary':
                    superlativeBinary_len += 1
                if datum['ans_type'] == 'open':
                    superlativeOpen_len += 1


            if datum['nc_dur'] == 1:
                duration_len += 1
                if datum['ans_type'] == 'binary':
                    durationBinary_len += 1
                if datum['ans_type'] == 'open':
                    durationOpen_len += 1


            if datum['nc_objrel'] == 1:
                object_relationship_len += 1
                if datum['ans_type'] == 'binary':
                    object_relationshipBinary_len += 1
                if datum['ans_type'] == 'open':
                    object_relationshipOpen_len += 1




            if ansString == label:
                overall += int(ansString == label)

                if datum['ans_type'] == 'binary':
                    allBinary += int(ansString == label)

                if datum['ans_type'] == 'open':
                    allOpen += int(ansString == label)


                if datum['nc_seq'] == 1:
                    sequencing += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        sequencingBinary += int(ansString == label)
                    if datum['ans_type'] == 'open':
                        sequencingOpen += int(ansString == label)

                if datum['nc_sup'] == 1:
                    superlative += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        superlativeBinary += int(ansString == label)
                    if datum['ans_type'] == 'open':
                        superlativeOpen += int(ansString == label)

                if datum['nc_dur'] == 1:
                    duration += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        durationBinary += int(ansString == label)
                    if datum['ans_type'] == 'open':
                        durationOpen += int(ansString == label)

                if datum['nc_objrel'] == 1:
                    object_relationship += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        object_relationshipBinary += int(ansString == label)
                    if datum['ans_type'] == 'open':
                        object_relationshipOpen += int(ansString == label)


        return [overall / len(quesid2ans), allBinary/allBinary_len, allOpen/allOpen_len, sequencing/sequencing_len,
                sequencingBinary/sequencingBinary_len, sequencingOpen/sequencingOpen_len,
                superlative/superlative_len, superlativeBinary/superlativeBinary_len, superlativeOpen/superlativeOpen_len,
                duration/duration_len, durationBinary/durationBinary_len, durationOpen/durationOpen_len,
                object_relationship/object_relationship_len, object_relationshipBinary/object_relationshipBinary_len,
                object_relationshipOpen/object_relationshipOpen_len]





    def evaluatePrecision(self, questions):
        objectPrecision = 0.
        objectPrecisionBinary = 0.
        objectPrecisionOpen = 0.

        objectPrecision_len = 0
        objectPrecisionBinary_len = 0
        objectPrecisionOpen_len = 0

        actionPrecision = 0.
        actionPrecisionBinary = 0.
        actionPrecisionOpen = 0.


        actionPrecision_len = 0
        actionPrecisionBinary_len = 0
        actionPrecisionOpen_len = 0


        localizationPrecision = 0.
        localizationPrecisionBinary = 0.
        localizationPrecisionOpen = 0.


        localizationPrecision_len = 0
        localizationPrecisionBinary_len = 0
        localizationPrecisionOpen_len = 0



        for q in questions:
            label = q["answer"]  # ground-truth answer
            prediction = q["prediction"]

            # length of each question type
            if q['i_obj'] == 1:
                objectPrecision_len += 1
                if q['ans_type'] == 'binary':
                    objectPrecisionBinary_len += 1
                if q['ans_type'] == 'open':
                    objectPrecisionOpen_len += 1

            if q['i_act'] == 1:
                actionPrecision_len += 1
                if q['ans_type'] == 'binary':
                    actionPrecisionBinary_len += 1
                if q['ans_type'] == 'open':
                    actionPrecisionOpen_len += 1

            if q['i_temp'] == 1:
                localizationPrecision_len += 1
                if q['ans_type'] == 'binary':
                    localizationPrecisionBinary_len += 1
                if q['ans_type'] == 'open':
                    localizationPrecisionOpen_len += 1

            # If correct answer
            if prediction == label:
                if q['i_obj'] == 1:
                    objectPrecision += 1
                    if q['ans_type'] == 'binary':
                        objectPrecisionBinary += 1
                    if q['ans_type'] == 'open':
                        objectPrecisionOpen += 1

                if q['i_act'] == 1:
                    actionPrecision += 1
                    if q['ans_type'] == 'binary':
                        actionPrecisionBinary += 1
                    if q['ans_type'] == 'open':
                        actionPrecisionOpen += 1

                if q['i_temp'] == 1:
                    localizationPrecision += 1
                    if q['ans_type'] == 'binary':
                        localizationPrecisionBinary += 1
                    if q['ans_type'] == 'open':
                        localizationPrecisionOpen += 1


        return [objectPrecision / objectPrecision_len, objectPrecisionBinary / objectPrecisionBinary_len,
        objectPrecisionOpen / objectPrecisionOpen_len,

        actionPrecision / actionPrecision_len, actionPrecisionBinary / actionPrecisionBinary_len,
        actionPrecisionOpen / actionPrecisionOpen_len,

        localizationPrecision / localizationPrecision_len, localizationPrecisionBinary / localizationPrecisionBinary_len,
        localizationPrecisionOpen / localizationPrecisionOpen_len]




    def evaluateIndirectRef(self, quesid2ans: dict):
        objectRecall = 0.
        objectRecallBinary = 0.
        objectRecallOpen = 0.

        objectRecall_len = 0
        objectRecallBinary_len = 0
        objectRecallOpen_len = 0


        actionRecall = 0.
        actionRecallBinary = 0.
        actionRecallOpen = 0.


        actionRecall_len = 0
        actionRecallBinary_len = 0
        actionRecallOpen_len = 0


        localizationRecall = 0.
        localizationRecallBinary = 0.
        localizationRecallOpen = 0.


        localizationRecall_len = 0
        localizationRecallBinary_len = 0
        localizationRecallOpen_len = 0

        precisionQs = []


        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['answer']  # ground-truth answer

            ansString = self.index_to_ans[ans]  # convert index to answer

            # length of each question type
            if datum['i_obj'] == 1:
                objectRecall_len += 1
                if datum['ans_type'] == 'binary':
                    objectRecallBinary_len += 1
                if datum['ans_type'] == 'open':
                    objectRecallOpen_len += 1


            if datum['i_act'] == 1:
                actionRecall_len += 1
                if datum['ans_type'] == 'binary':
                    actionRecallBinary_len += 1
                if datum['ans_type'] == 'open':
                    actionRecallOpen_len += 1


            if datum['i_temp'] == 1:
                localizationRecall_len += 1
                if datum['ans_type'] == 'binary':
                    localizationRecallBinary_len += 1
                if datum['ans_type'] == 'open':
                    localizationRecallOpen_len += 1


            # If correct answer
            if ansString == label:
                if datum['i_obj'] == 1:
                    objectRecall += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        objectRecallBinary += int(ansString == label)
                    if datum['ans_type'] == 'open':
                        objectRecallOpen += int(ansString == label)

                if datum['i_act'] == 1:
                    actionRecall += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        actionRecallBinary += int(ansString == label)
                    if datum['ans_type'] == 'open':
                        actionRecallOpen += int(ansString == label)

                if datum['i_temp'] == 1:
                    localizationRecall += int(ansString == label)
                    if datum['ans_type'] == 'binary':
                        localizationRecallBinary += int(ansString == label)
                    if datum['ans_type'] == 'open':
                        localizationRecallOpen += int(ansString == label)



            # See if same question with a direct reference was answered correctly
            if datum['direct_equiv'] != None and datum['indirect'] == 1:
                equivalentQ = datum['direct_equiv']  # questionID

                # If the question ID exists in the dataset
                if equivalentQ in self.dataset.id2datum:
                    equivalentData = self.dataset.id2datum[equivalentQ]

                    equivalentAns = equivalentData['answer']  # ground-truth answer
                    equivalentPrediction = quesid2ans[equivalentQ]  # predicted answer
                    equivalentString = self.index_to_ans[equivalentPrediction]  # answer as a string


                    # If correct answer, add indirect reference question to list.
                    # Pass to evaluatePrecision() later
                    if equivalentString == equivalentAns:
                        datum["prediction"] = ansString
                        precisionQs.append(datum)





        return [objectRecall / objectRecall_len,  objectRecallBinary/objectRecallBinary_len, objectRecallOpen/objectRecallOpen_len,
                actionRecall / actionRecall_len, actionRecallBinary/actionRecallBinary_len, actionRecallOpen/actionRecallOpen_len,

                localizationRecall / localizationRecall_len, localizationRecallBinary/localizationRecallBinary_len,
                localizationRecallOpen/ localizationRecallOpen_len], precisionQs









    def save_json(self, data, file_path):
        with open(file_path, "w") as f:
            json.dump(data, f)
            f.close()



    def dump_result(self, quesid2ans: dict, path):
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                prediction = self.index_to_ans[ans]

                datum = self.dataset.id2datum[ques_id]
                label = datum['answer']

                if args.indirect_ref:
                    result.append({
                        'id': datum['question_id'],
                        'question': datum['question'],
                        'ans_type': datum['ans_type'],
                        'question type': datum['global'],
                        'prediction': prediction,
                        'answer': label,
                        'directEq': datum['direct_equiv'],
                        'i_obj': datum['i_obj'],
                        'i_act': datum['i_act'],
                        'i_temp': datum['i_temp'],
                        'indirectFlag': datum['indirect']
                    })

                else:
                    result.append({
                        'id': datum['question_id'],
                        'question': datum['question'],
                        'ans_type': datum['ans_type'],
                        'question type': datum['global'],
                        'prediction': prediction,
                        'answer': label,
                        'steps:': datum['steps'],
                        'more_steps': datum['more_steps']
                    })

            json.dump(result, f, indent=4, sort_keys=True)
