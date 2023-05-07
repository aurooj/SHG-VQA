# coding=utf-8
# Copyleft 2019 Project LXRT
import json
import sys
import csv
import base64
import time

import h5py
# import pandas as pd
import pickle
import os

import numpy as np

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

def save_pickle(data, data_path):
    with open(data_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)

    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                if item[key] == '':
                    item[key] = 0
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (7, 7, 1024), np.float64),
            ]
            for key, shape, dtype in decode_config:
                if key == 'features':
                    decoded_item = base64.b64decode(item[key])
                    item[key] = np.frombuffer(decoded_item) #todo: replace dummy data with orig features
                else:
                    item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


def load_spatial_data(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    fparts = fname.split('/')
    print(fparts[:-1])
    fpath = os.path.join(*fparts[:-1])
    fn = fparts[-1]
    split = fn.split('_')[0]
    mapping_fn = os.path.join(  fpath, 'img_id2idx_{}.json'.format(split))
    start_time = time.time()
    print("Reading %s file"% mapping_fn)
    img_id2idx_dict = load_json(mapping_fn)
    print("Start to load ResNet152 features from %s" % fname)

    h = h5py.File(os.path.join( fpath, '{}_features.hdf5'.format(split)), 'r')
    img_features = h['data']


    for img_id, item in img_id2idx_dict.items():
        item["features"] = img_features[item["i"]]
        item["img_id"] = img_id

        for key in ['objects_id', 'objects_conf', 'attrs_id', 'attrs_conf', 'boxes', 'features']:
            if item[key] is not None:
                item[key].setflags(write=False)
            else:
                if key == 'boxes':
                    item[key] = np.zeros((1,4))
                else:
                    item[key] = np.array([0,0])


        data.append(item)

        if topk is not None and len(data) == topk:
            break

    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    #return topK results
    # if topk != -1 or topk != None:
    #     data = data[:topk]
    return data


def load_spatial_gqa(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    #todo: adopt function to read gqa data
    data = []
    fparts = fname.split('/')
    print(fparts[:-1])
    fpath = os.path.join(*fparts[:-1])
    fn = fparts[-1]
    split = fn.split('_')[0]
    mapping_fn = os.path.join( fpath, 'gqa_spatial_merged_info.json')
    start_time = time.time()
    print("Reading %s file"% mapping_fn)
    img_id2idx_dict = load_json(mapping_fn)
    print("Start to load ResNet152 features from %s" % fname)

    h = h5py.File(os.path.join(fpath, 'gqa_spatial.h5'), 'r')
    img_features = h['features']


    for img_id, item in img_id2idx_dict.items():
        item["features"] = img_features[item["index"]]
        item["img_id"] = img_id

        item['objects_id'] = None
        item['objects_conf'] = None
        item['attrs_id'] = None
        item['attrs_conf'] = None
        item['boxes'] = None
        item['num_boxes'] = 0

        for key in ['objects_id', 'objects_conf', 'attrs_id', 'attrs_conf', 'boxes', 'features']:
            if item[key] is not None:
                item[key].setflags(write=False)
            else:
                if key == 'boxes':
                    item[key] = np.zeros((1,4))
                else:
                    item[key] = np.array([0,0])


        data.append(item)

        if topk is not None and len(data) == topk:
            break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

def load_patches(fname, dataset='',topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    #todo: adopt function to read gqa data
    assert dataset != ''
    data = []
    fparts = fname.split('/')
    print(fparts[:-1])
    fpath = os.path.join(*fparts[:-1])
    fn = fparts[-1]
    split = fn.split('_')[0]
    mapping_fn =  os.path.join(fpath, 'img_id2idx_{dataset}_{split}_32x32.json'.format(dataset=dataset,
                                                                                            split=split))
    start_time = time.time()
    print("Reading %s file"% mapping_fn)
    img_id2idx_dict = load_json(mapping_fn)
    print("Start to load image patches from %s" % fname)

    h = h5py.File(os.path.join(fpath, '{split}_patches_32x32.hdf5'.format(split=split)), 'r')
    img_features = h['data']


    for img_id, item in img_id2idx_dict.items():
        item["features"] = img_features[item["i"]]
        item["img_id"] = img_id

        item['objects_id'] = None
        item['objects_conf'] = None
        item['attrs_id'] = None
        item['attrs_conf'] = None
        item['boxes'] = None
        item['num_boxes'] = 0

        for key in ['objects_id', 'objects_conf', 'attrs_id', 'attrs_conf', 'boxes', 'features']:
            if item[key] is not None:
                item[key].setflags(write=False)
            else:
                if key == 'boxes':
                    item[key] = np.zeros((1,4))
                else:
                    item[key] = np.array([0,0])


        data.append(item)
        if topk is not None and len(data) == topk:
            break

    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


