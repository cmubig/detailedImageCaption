import tensorflow as tf
from tensorpack import *

# -*- coding: utf-8 -*-
# File: box_ops.py

import tensorflow as tf
from tensorpack.tfutils.scope_utils import under_name_scope

"""
This file is modified from
https://github.com/tensorflow/models/blob/master/object_detection/core/box_list_ops.py
"""
def offset(boxlist1,boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: Nx1x4 floatbox
      boxlist2: NxDx4

    Returns: Nx2*D [x,x,x,y,y,y,]   
    """
    num_iou = tf.shape(boxlist2)[1]
    boxlist1 = tf.tile(boxlist1,[1,num_iou,1]) # NxDx4
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=2) # NxDx1
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=2) # NxDx1

    c_x1 = (x_max1 - x_min1) /2 + x_min1
    c_y1 = (y_max1 - y_min1) /2 + y_min1
    c_x2 = (x_max2 - x_min2) /2 + x_min2
    c_y2 = (y_max2 - y_min2) /2 + y_min2

    return tf.reshape(tf.concat([c_x1-c_x2,c_y1-c_y2],1),[-1,2*num_iou]) # Nx2*D


def area(boxes):
    """
    Args:
      boxes: NxDx4 floatbox

    Returns:
      n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=2) # NxDx1
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [2]) # NxD


def pairwise_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: Nx1x4 floatbox
      boxlist2: NxDx4

    Returns:    
    """
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=2) # NxDx1
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=2) # NxDx1
    all_pairs_min_ymax = tf.minimum(y_max1, y_max2)
    all_pairs_max_ymin = tf.maximum(y_min1, y_min2)
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)#  NxDx1
    all_pairs_min_xmax = tf.minimum(x_max1, x_max2)
    all_pairs_max_xmin = tf.maximum(x_min1, x_min2)
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin) # NxDx1
    return tf.squeeze(intersect_heights * intersect_widths,[2]) # NxD


def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: Nx1x4 floatbox
      boxlist2: NxDx4 D: # of referetial objects

    Returns:
      a tensor with shape [N, D] representing pairwise iou scores.
    """
    num_iou = tf.shape(boxlist2)[1]
    boxlist1 = tf.tile(boxlist1,[1,num_iou,1]) # NxDx4

    intersections = pairwise_intersection(boxlist1, boxlist2) # NxD
    areas1 = area(boxlist1) # NxD
    areas2 = area(boxlist2) # NxD
    unions = areas1 + areas2 - intersections # NxD
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions)) # NxD
