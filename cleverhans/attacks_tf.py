from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings

from . import utils_tf

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS



def fgm_all_feats(x, src_x,last_grad,preds,src_all_feats,dist_all_feats,
        feat_scale, conv_scale,pow_scale,
        y=None, eps=0.3, ord=np.inf, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor
    :param y: (optional) A placeholder for the model labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    feat_loss = 0
    for feats in zip(src_all_feats,dist_all_feats):
      feat_loss += tf.reduce_mean(tf.nn.l2_loss(tf.sqrt(tf.abs(feats[0]))*tf.sign(feats[0])-tf.sqrt(tf.abs(feats[1]))*tf.sign(feats[1])))
      #feat_loss += tf.reduce_mean(tf.nn.l2_loss(tf.pow(tf.abs(feats[0]),tf.ones(feats[0].get_shape())*pow_scale)*tf.sign(feats[0])-
      #    tf.pow(tf.abs(feats[1]),tf.ones(feats[0].get_shape())*pow_scale)*tf.sign(feats[1])))
    feat_loss = feat_loss / len(src_all_feats) 
    softmax_loss = tf.reduce_mean(utils_tf.model_loss(y, preds, mean=False))
    filter_p = tf.constant(np.ones((3,3,3,1))/(9.0*3.0),dtype=np.float32)
    conv_loss = tf.reduce_mean(tf.abs(tf.nn.conv2d(x-src_x,filter_p,strides=[1,1,1,1],padding='SAME',data_format='NHWC')))

    loss = softmax_loss + feat_scale * feat_loss + conv_scale * conv_loss

    # Define gradient of loss wrt input
    #adv_last_feat = tf.stop_gradient(adv_last_feat)
    grad, = tf.gradients(loss, x)
    if(last_grad is not None):
      grad = grad*0.3 + last_grad*0.7
    grad = tf.where(tf.is_nan(grad), tf.random_normal(grad.get_shape())*0.01, grad)
    

    if ord == np.inf:
        # Take sign of gradient
        signed_grad = tf.sign(grad)
    elif ord == 1:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.reduce_sum(tf.abs(grad),
                                           reduction_indices=reduc_ind,
                                           keep_dims=True)
    elif ord == 2:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad),
                                                   reduction_indices=reduc_ind,
                                                   keep_dims=True))
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad
    #if(last_grad is not None):
    #  scaled_signed_grad = scaled_signed_grad*0.3 + last_grad*0.7

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x,grad
