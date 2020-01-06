#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import matplotlib.pyplot as plt
import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import numpy as np
import json
import six
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

assert six.PY3, "FasterRCNN requires Python 3!"

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_tuple
import tensorpack.utils.viz as tpviz

from coco import COCODetection
from basemodel import (
    image_preprocess, resnet_c4_backbone, resnet_conv5,
    resnet_fpn_backbone)

import model_frcnn
import model_mrcnn
from model_frcnn import (
    sample_fast_rcnn_targets, fastrcnn_outputs,
    fastrcnn_predictions, BoxProposals, FastRCNNHead)
from model_mrcnn import maskrcnn_upXconv_head, maskrcnn_loss
from model_rpn import rpn_head, rpn_losses, generate_rpn_proposals
from model_fpn import (
    fpn_model, multilevel_roi_align,
    multilevel_rpn_losses, generate_fpn_proposals)
from model_cascade import CascadeRCNNHead
from utils.model_box import (
    clip_boxes, crop_and_resize, roi_align, RPNAnchors)

from data import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors, get_all_anchors_fpn)
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs)
from eval import (
    eval_coco, detect_one_image, print_evaluation_scores, DetectionResult)
from config import finalize_configs, config as cfg
from utils.common import CustomResize, clip_boxes
import progressbar
import pickle
class DetectionModel(ModelDesc):
    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

class ResNetC4Model(DetectionModel):
    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.float32, (10, 4), 'boxes')]
        return ret

    def build_graph(self, *inputs):
        # TODO need to make tensorpack handles dict better
        inputs = dict(zip(self.input_names, inputs))
        image = self.preprocess(inputs['image'])     # 1CHW
        box = inputs['boxes'] 
        region_mask =  tf.to_float(tf.not_equal(tf.reduce_sum(box,axis=1,keep_dims=True),0.0)) # 10 x 1

        
        #featuremap = resnet_c4_backbone(image, [3,4,23])
        #temp = resnet_conv5(featuremap, 3)
        #global_feature = GlobalAvgPooling('global_feature', temp, data_format='channels_first') #(1,2048)

        #boxes_on_featuremap = box * (1.0 / 16)
        #roi_resized = roi_align(featuremap, box, 14)
        #temp = resnet_conv5(roi_resized, 3)    # nxcx7x7
        

        c2345 = resnet_fpn_backbone(image, [3,4,23,3])
        global_feature = GlobalAvgPooling('global_feature', c2345[-1], data_format='channels_first') #(1,2048)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES,global_feature)

        p23456 = fpn_model('fpn', c2345)
        roi_feature_fastrcnn = multilevel_roi_align(p23456[:4],box, 7)
        region_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn)
        region_feature = GlobalAvgPooling('gap',temp, data_format='channels_first') #(10,2048)
        region_feature = tf.multiply(region_feature,region_mask,name='region_feature')
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES,region_feature)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--mode',help='val or train')
    parser.add_argument('--image_dir')
    parser.add_argument('--save_dir')
    
    args = parser.parse_args()
    assert args.load
    if args.mode == 'val':
        box = pickle.load(open('val2014_box.pkl','rb'))
    else:
        box = pickle.load(open('train2014_box.pkl','rb'))
    
    MODEL = ResNetC4Model()
    


    input_tensor = ['image','boxes']
    output_tensor = ['global_feature','region_feature']

    
    pred = OfflinePredictor(PredictConfig(
        model=MODEL,
        session_init=get_model_loader(args.load),
        input_names=input_tensor,
        output_names=output_tensor))

    files = os.listdir(args.image_dir)
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i,file in enumerate(files):
            file_name = os.path.join(args.image_dir,file)
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            #import ipdb;ipdb.set_trace() 
            #orig_shape = img.shape[:2]
            resizer = CustomResize(800, 1333) #[600,800]
            resized_img = resizer.augment(img)
            scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
            global_feature,region_feature = pred(resized_img,box[i]*scale)#(1,2048) (10,2048)
            features = [global_feature,region_feature]
            import ipdb;ipdb.set_trace()
            file_name = os.path.join(args.save_dir,file.split('.')[0]+'.pkl')
            pickle.dump(features,open(file_name,'wb'))
            bar.update(i)
        #import ipdb;ipdb.set_trace()










