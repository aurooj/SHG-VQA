# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/star')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    parser.add_argument("--vitInit", dest='vit_init', action='store_const', default=False, const=True,
                        help='If --vitInit specified, rlayers will be initialized from vit weights '
                             'starting from layer index specified with --startIndex')



    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)
    parser.add_argument("--noObjWeight", dest='no_object_weight', default=0.1, type=float)
    parser.add_argument("--logFreq", dest='log_freq', default=50, type=int)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')
    parser.add_argument("--dlayers", default=5, type=int, help='Number of relation decoder layers.')
    parser.add_argument("--startIndex", dest='start_index', default=7, type=int, help='Specify the layer index to start loading vit weights from.')
    parser.add_argument("--skipConnection", dest='skip_connection', action='store_const', default=False, const=True)
    parser.add_argument("--sharedWeights", dest='shared_weights', action='store_const', default=False, const=True)
    parser.add_argument("--normInputs", dest='norm_inputs', action='store_const', default=False, const=True)
    parser.add_argument("--crossAttn", dest='cross_attn', action='store_const', default=False, const=True)
    parser.add_argument("--crossAttnType", dest='cross_attn_type', default="cross", type=str,
                        choices=["cross", "self", 'cross_self', 'no_cross', 'old'], help='Types of cross-modality attention')
    parser.add_argument("--patches", dest='patches', action='store_const', default=False, const=True)
    parser.add_argument("--attnRouting", dest='attn_routing', action='store_const', default=False, const=True)
    parser.add_argument("--freezeWeights", dest='freeze_weights', action='store_const', default=False, const=True)
    parser.add_argument("--noCaps", dest='no_caps', action='store_const', default=False, const=True)
    parser.add_argument("--NUM_PRIM_CAPS", default=32, type=int, help='Number of primary capsules.')
    parser.add_argument("--NUM_VIS_CAPS", default=32, type=int, help='Number of visual capsules.')
    parser.add_argument("--POSE_DIM", default=4, type=int, help='Pose matrix size. Default is 4.')
    parser.add_argument("--HW", default=7, type=int, help='Spatial feature map size.')

    #LXRT evaluation
    parser.add_argument("--outputAttn", dest='output_attention', action='store_const', default=False, const=True)

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--taskGrounding", dest='task_grounding', action='store_const', default=False, const=True)
    parser.add_argument("--taskContrastive", dest='task_contrastive', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--excludeSet", dest='exclude_set', default='', type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    #AGQA dataset config
    parser.add_argument("--qType", dest='qtype', default="Feasibility", type=str,
                        choices=["Feasibility", "Prediction", 'Interaction', 'Sequence'],
                        help='Types of question to train model individually on.')
    parser.add_argument("--mergeData", dest='merge_data', action='store_const', default=False, const=True,
                        help='If true, train a single model on all question types filtering out questions.')
    parser.add_argument("--mergeAll", dest='merge_all', action='store_const', default=False, const=True,
                        help='If true, train a single model on all question types without filtering out data.')
    parser.add_argument("--qaArrangeType", dest='qa_arrange_type', default="add_sep_all", type=str,
                        choices=['add_sep_all', 'no_sep_all', 'add_sep', 'no_sep'],
                        help="Different ways to arrange Question and Answer choices ")
    parser.add_argument("--numRel", dest="num_rel", default=8, type=int, help='Maximum number of relations in hypergraph. Default is 8')
    parser.add_argument("--numAct", dest="num_act", default=3, type=int,
                        help='Maximum number of actions in hypergraph. Default is 3')
    parser.add_argument("--addAction", dest='add_action', action='store_const', default=False, const=True)
    parser.add_argument("--addRelation", dest='add_relation', action='store_const', default=False, const=True)
    parser.add_argument("--numSituations", dest="num_situations", default=16, type=int,
                        help='Maximum number of relations in hypergraph. Default is 16')
    parser.add_argument("--clipLEN", dest="CLIP_LEN", default=16, type=int,
                        help='Number of frames in the input video. Default is 16')
    parser.add_argument("--trainSubSet", dest="train_sub_set", action='store_const', default=False, const=True,
                        help='Load training sub set')



    #data processing config
    parser.add_argument("--augmentType", dest='augment_type', default="no_aug", type=str,
                        choices=["no_aug", "no_aug_slowfast", "aug_mix", 'rand_aug', "rand_aug_slowfast"],
                        help="Augmentation types for video. At test time, we use 'no_aug'")

    #AGQA model config
    parser.add_argument("--afterCrossAttnFeats", dest='after_cross_attn_feats', action='store_const', default=False, const=True)
    parser.add_argument("--linearCls", dest='linear_cls', action='store_const', default=False,
                        const=True)
    parser.add_argument("--embDropRate", dest='emb_drop_rate', default=0.15, type=float)
    parser.add_argument("--decoderDropRate", dest='decoder_drop_rate', default=0.15, type=float)
    parser.add_argument("--taskQ", dest='task_q', action='store_const', default=False, const=True)
    parser.add_argument("--taskVQA", dest='task_vqa', action='store_const', default=False, const=True)
    parser.add_argument("--taskHGQA", dest='task_hgqa', action='store_const', default=False, const=True)
    parser.add_argument("--taskVHGA", dest='task_vhga', action='store_const', default=False, const=True)
    parser.add_argument("--taskHGVQA", dest='task_hgvqa', action='store_const', default=False, const=True,
                        help='if true, cross attentional module takes Q, V, and HG.')
    parser.add_argument("--GTHG", dest='gt_hg', action='store_const', default=False, const=True)
    parser.add_argument("--useHGMask", dest='use_hg_mask', action='store_const', default=False,
                        const=True)
    parser.add_argument("--LossHGPerFrame", dest='loss_hg_per_frame', action='store_const', default=False, const=True)
    
    
    
    #AGQA test splits
    parser.add_argument("--novelComp", dest='novel_comp', action='store_const', default=False, const=True)
    parser.add_argument("--indirectRef", dest='indirect_ref', action='store_const', default=False, const=True)
    parser.add_argument("--compSteps", dest='comp_steps', action='store_const', default=False, const=True)
    


    # backbone
    parser.add_argument('--backbone',
                        dest='backbone',
                        default='slow_r50',
                        const='slow_r50',
                        nargs='?',
                        choices=['slow_r50', 'slowfast_r50', 'slowfast_r101', 'resnext101', 'video_swin', 'mvit_B'],
                        help='backbones for video features.')
    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=8)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)


    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()

def action_prediction_base_classifier(args):
    args.classifier = 'base'
    return args