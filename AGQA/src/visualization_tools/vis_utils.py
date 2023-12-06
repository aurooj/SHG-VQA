import json
import os
import pickle

import numpy as np
import cv2
import ipyplot
import matplotlib.pyplot as plt
import src.utils as utils


def sample_frames(frame_ids, max_show_num):
    # sample frames from given frame IDs averagely according to max_show_num
    if max_show_num == 0:
        return frame_ids
    max_show_num = min(len(frame_ids), max_show_num)
    interval = int(len(frame_ids) / max_show_num)
    return frame_ids[::interval]


def trim_keyframes(data, frameDict, max_show_num=4):
    vidId = data["video_id"]
    questionId = data["question_id"]

    # trimmed_frame_ids = totalGroundings[questionId]


    # if trimmed_frame_ids == []:     # If no grounding frames were given, take all keyframes and sample
    trimmed_frame_ids = frameDict[vidId]


    trimmed_frame_ids = list(sorted(trimmed_frame_ids))
    trimmed_frame_ids = sample_frames(trimmed_frame_ids, max_show_num)
    return trimmed_frame_ids


# def trim_keyframes(data, fps, max_show_num=4):
#     frame_ids = list(sorted(data['situations'].keys()))
#     trimmed_frame_ids = [frame for frame in frame_ids if
#                          int(frame) >= (data['start']) * fps[data['video_id'] + '.mp4'] + 1 and int(frame) < (
#                          data['end']) * fps[data['video_id'] + '.mp4'] + 1]
#     trimmed_frame_ids = sample_frames(trimmed_frame_ids, max_show_num)
#     return trimmed_frame_ids

#
# def trim_keyframes_test(data, frame_ids, fps, max_show_num=4):
#     trimmed_frame_ids = [frame for frame in frame_ids if
#                          int(frame) >= (data['start']) * fps[data['video_id'] + '.mp4'] + 1 and int(frame) < (
#                          data['end']) * fps[data['video_id'] + '.mp4'] + 1]
#     trimmed_frame_ids = sample_frames(trimmed_frame_ids, max_show_num)
#     return trimmed_frame_ids


def frame_plot(frame_list, frame_dir):
    select = []
    for i in range(len(frame_list)):
        frame = cv2.imread(frame_dir + '/' + frame_list[i] + '.png')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        select.append(frame)
    ipyplot.plot_images(select, max_images=len(select), img_width=150)


def vis_keypoints(img, kpts):
    link2 = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1)]
    x_ = kpts[0::3]
    y_ = kpts[1::3]
    v_ = kpts[2::3]
    x_max = np.float32(max(x_))
    x_min = np.float32(min(x_))
    y_max = np.float32(max(y_))
    y_min = np.float32(min(y_))
    x_len = x_max - x_min
    y_len = y_max - y_min
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(link2) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for i in range(len(link2)):
        order1, order2 = link2[i][0], link2[i][1]
        x1 = int(np.float32(x_[order1]))
        y1 = int(np.float32(y_[order1]))
        x2 = int(np.float32(x_[order2]))
        y2 = int(np.float32(y_[order2]))

        if v_[order1] > 0 and v_[order2] > 0:
            cv2.line(img, (x1, y1), (x2, y2), thickness=4, color=colors[i])
    i = 0
    for x, y, v in zip(x_, y_, v_):
        x = int(np.float32(x))
        y = int(np.float32(y))
        if v > 0:
            img = cv2.circle(
                img, (x, y), 1, (0, 0, 255), -1)
            i = i + 1

    return img


def group_by_vid(QA):
    qa_by_vid = {}
    vid = {}
    for qa in QA:
        if qa['video_id'] not in qa_by_vid:
            qa_by_vid[qa['video_id']] = [qa]
        else:
            qa_by_vid[qa['video_id']].append(qa)
    return qa_by_vid


def split_qtypes_in_vid(QA):
    for vid in QA:
        QA[vid] = group_by_qtypes(QA[vid])

    return QA

#### custom functions by Aisha ##########
def group_by_qtypes(QA):
    qa_by_qtype = {
        "Interaction": [],
        "Sequence": [],
        "Prediction": [],
        "Feasibility": []
    }
    for qa in QA:
        qa_by_qtype[qa['question_id'].split('_')[0]].append(qa)

    return qa_by_qtype

def get_merged_data(data):
    video_ids = set()
    for dt in data:

        video_ids.add(dt['video_id'])

    qa_qtype = group_by_qtypes(data)

    vid_by_qtypes = {}
    for qtype, datum in qa_qtype.items():
        video_ids = set()
        for dt in datum:
            video_ids.add(dt['video_id'])
        vid_by_qtypes[qtype] = video_ids

    #find the set of videos which are common in merged sets of Interaction, Sequence and Prediction, Feasibility,
    # because we want to exclude those videos and related question types from Interaction and Sequence in order to
    # avoid data leakage for Prediction and Feasibility questions in training data.
    vid_ids_ = (vid_by_qtypes['Interaction'].union(vid_by_qtypes['Sequence'])).intersection(
        vid_by_qtypes['Prediction'].union(vid_by_qtypes['Feasibility']))

    filtered_qa_qtypes = {}
    for qtype, datum in qa_qtype.items():
        filtered_datum = [dt for dt in datum if dt['video_id'] not in vid_ids_]
        if qtype in ['Interaction', 'Sequence']:
            filtered_qa_qtypes[qtype] = filtered_datum
        else:
            filtered_qa_qtypes[qtype] = datum

    return filtered_qa_qtypes

#######################################

def select_by_vid(QA, vid):
    qa_select = []
    if vid == '':
        return QA

    for qa in QA:
        if qa['video_id'] == vid:
            qa_select.append(qa)

    if qa_select == []:
        return QA

    return qa_select


def select_by_qid(QA, qid):
    selected_qa = []

    for qa in QA:
        if qa['question_id'] in qid:
            selected_qa.append(qa)

    if len(selected_qa) == 0:
        return QA

    return selected_qa


def get_vocab(label_dir):
    obj_to_ind, ind_to_obj, ind_to_rel, rel_to_ind = {}, {}, {}, {}
    obj_vocab, rel_vocab, verb_vocab = [], [], []

    with open(label_dir + "/object_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.strip('\n')
            obj_vocab.append(mapping.split(' ')[1])

    with open(label_dir + "/relationship_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.strip('\n')
            rel_vocab.append(mapping.split(' ')[1])


    with open(label_dir + "/verb_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.strip('\n')
            rel_vocab.append(mapping.split(' ')[1])

    return obj_vocab, rel_vocab, verb_vocab


def get_act_cls(label_dir):
    dict = {}
    with open(label_dir + "/action_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            tag = line[0:4]
            description = line[5:-1]
            dict[tag] = description
    return dict


def get_vocab_dict(label_dir):
    obj_to_ind, ind_to_obj, ind_to_rel, rel_to_ind = {}, {}, {}, {}
    obj_vocab, rel_vocab, verb_vocab = {}, {}, {}

    with open(label_dir + "/object_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.strip('\n')
            oitem = mapping.split(' ')
            obj_vocab[oitem[0]] = oitem[1]

    with open(label_dir + "/relationship_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.strip('\n')
            ritem = mapping.split(' ')
            rel_vocab[ritem[0]] = ritem[1]

    with open(label_dir + "/verb_classes.txt") as f:
        lines = f.readlines()
        for line in lines:
            mapping = line.strip('\n')
            ritem = mapping.split(' ')
            verb_vocab[ritem[0]] = ritem[1]

    return obj_vocab, rel_vocab, verb_vocab


def get_scene_graphs(label_dir):
    f = open(label_dir, 'rb')
    sceneGraphs = pickle.load(f)

    return sceneGraphs


def get_answer_vocab(label_dir):
    f = open(label_dir, 'rb')
    ansVocab = pickle.load(f)

    return ansVocab




def create_relationship_data(annotation_dir, sceneGraphs):
    obj_vocab, rel_vocab, verb_vocab = get_vocab_dict(annotation_dir)

    rel_triplets = []
    rel_triplet_labels = []

    # For each videoID, only get their frames
    for videoId, val in sceneGraphs.items():
        for key, typ in val.items():
            if typ['type'] == 'frame':
                objList = sceneGraphs[videoId][key]['objects']['names']  # list of all objs in the current frame

                # format to 'obj/000025'
                for obj in objList:
                    formatFrame = obj + '/' + key

                    # get relationships for current obj
                    spatial = sceneGraphs[videoId][formatFrame]['spatial']
                    contact = sceneGraphs[videoId][formatFrame]['contact']
                    attention = sceneGraphs[videoId][formatFrame]['attention']
                    verb = sceneGraphs[videoId][formatFrame]['verb']

                    PERSON = 'o1'
                    currObj = obj

                    # if relationships exist, make a triplet
                    if spatial != []:
                        for rel in spatial:
                            rel_tuple = (PERSON, rel['class'], currObj)
                            if rel_tuple not in rel_triplets:
                                rel_triplets.append(rel_tuple)
                                rel_triplet_labels.append(
                                    (obj_vocab[PERSON], rel_vocab[rel['class']], obj_vocab[currObj]))

                    if contact != []:
                        for rel in contact:
                            rel_tuple = (PERSON, rel['class'], currObj)
                            if rel_tuple not in rel_triplets:
                                rel_triplets.append(rel_tuple)
                                rel_triplet_labels.append(
                                    (obj_vocab[PERSON], rel_vocab[rel['class']], obj_vocab[currObj]))

                    if attention != []:
                        for rel in attention:
                            rel_tuple = (PERSON, rel['class'], currObj)
                            if rel_tuple not in rel_triplets:
                                rel_triplets.append(rel_tuple)
                                rel_triplet_labels.append(
                                    (obj_vocab[PERSON], rel_vocab[rel['class']], obj_vocab[currObj]))

                    if verb != []:
                        for rel in verb:
                            rel_tuple = (PERSON, rel['class'], currObj)
                            if rel_tuple not in rel_triplets:
                                rel_triplets.append(rel_tuple)
                                rel_triplet_labels.append(
                                    (obj_vocab[PERSON], verb_vocab[rel['class']], obj_vocab[currObj]))



    print(f"Number of unique relationship triplets:{len(rel_triplets)}/{len(rel_triplet_labels)}")
    rel_class_idxes = list(range(1, len(rel_triplets) + 1))
    rel_triplets_idx2rp = {k: v for k, v in zip(rel_class_idxes, rel_triplets)}
    rel_triplets_rp2idx = {k: v for k, v in zip(rel_triplets, rel_class_idxes)}
    rel_triplets_data = {'rel_triplets': list(rel_triplets),
                         'rel_triplets_lbls': rel_triplet_labels,
                         'rel_triplets_idx2rp': rel_triplets_idx2rp,
                         'rel_triplets_rp2idx': rel_triplets_rp2idx
                         }
    utils.save_pickle(rel_triplets_data, os.path.join(annotation_dir, 'relationship_triplets.json'))
    return rel_triplets_data





def get_action_dictionaries(annotation_dir):
    action_classes = get_act_cls(annotation_dir)
    act_class_idxes = list(range(1, len(action_classes) + 1))
    act_idx2rp = {k: v for k, v in zip(act_class_idxes, list(action_classes.keys()))}
    act_rp2idx = {k: v for k, v in zip(list(action_classes.keys()), act_class_idxes)}

    action_data = {'actions_idx2rp': act_idx2rp,
                   'actions_rp2idx': act_rp2idx
                   }
    utils.save_pickle(action_data, os.path.join(annotation_dir, 'action_dictionaries.json'))
    return action_data


# this function is modified from: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/transforms/functional.py
# all rights belongs to Facebook and its affiliates
import torch


def uniform_subsample(x: torch.Tensor, num_samples: int, temporal_dim: int = -3
                      ) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.
    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.
    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices), indices


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
