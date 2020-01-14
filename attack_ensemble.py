"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import OwnMethod
from AlexNet import AlexNet
import numpy as np
from PIL import Image

import tensorflow as tf

from nets import inception_resnet_v2
from nets import inception_v3
from nets import inception_v4
from nets import vgg
from nets import resnet_v2
from nets import inception_utils

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape,src_batch_shape):
  """Read png images from input directory in batches.
  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  src_images = np.zeros(src_batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      src_img = Image.open(f).convert('RGB')
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    src_images[idx, :, :, :] = np.array(src_img).astype(np.float) 
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, src_images
      filenames = []
      idx = 0
  if idx > 0:
    yield filenames, src_images

def save_images(images, filenames, output_dir):
  """Saves images to the output directory.
  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (images[i, :, :, :]).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')

class EnsembleModel(object):
  """Model class for CleverHans library."""
  def __init__(self, num_classes):
    self.num_classes = num_classes
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      self.network_fn_incep_res = inception_resnet_v2.inception_resnet_v2
    with slim.arg_scope(vgg.vgg_arg_scope()):
      self.network_fn_vgg16 = vgg.vgg_16
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      self.network_fn_res = resnet_v2.resnet_v2_152
    with slim.arg_scope(inception_utils.inception_arg_scope()):
      self.network_fn_incepv3 = inception_v3.inception_v3
      self.network_fn_incepv4 = inception_v4.inception_v4
    self.network_fn_alex = AlexNet()
    self.build = False
  def __call__(self, x_input):
    if(self.build):
      tf.get_variable_scope().reuse_variables()
    else:
      self.build = True
    inception_imags = (x_input/255.0-0.5)*2 
    resized_images_vgg = tf.image.resize_images(x_input,[224,224]) - tf.constant([123.68,116.78,103.94])
    with slim.arg_scope(vgg.vgg_arg_scope()):
      logits_vgg16, _ = self.network_fn_vgg16(resized_images_vgg,num_classes=self.num_classes,is_training=False)
    resized_images_res = (tf.image.resize_images(x_input,[224,224])/255.0-0.5)*2
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      logits_res, _ = self.network_fn_res(resized_images_res,num_classes=self.num_classes+1,is_training=False)
    logits_res = tf.reshape(logits_res,(-1,1001));
    logits_res = tf.slice(logits_res,[0,1],[FLAGS.batch_size,self.num_classes])
    with slim.arg_scope(inception_utils.inception_arg_scope()):
      logits_incepv3, _ = self.network_fn_incepv3(inception_imags,num_classes=self.num_classes+1,is_training=False)
    logits_incepv3 = tf.slice(logits_incepv3,[0,1],[FLAGS.batch_size,self.num_classes])
    with slim.arg_scope(inception_utils.inception_arg_scope()):
      logits_incepv4, _ = self.network_fn_incepv4(inception_imags,num_classes=self.num_classes+1,is_training=False)
    logits_incepv4 = tf.slice(logits_incepv4,[0,1],[FLAGS.batch_size,self.num_classes])
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      logits_incep_res,_ = self.network_fn_incep_res(inception_imags,num_classes=self.num_classes+1,is_training=False)
    logits_incep_res = tf.slice(logits_incep_res,[0,1],[FLAGS.batch_size,self.num_classes])
    alex_images = tf.image.resize_images(x_input,[256,256])
    alex_images = tf.reverse(alex_images,axis=[-1])
    alex_mean_npy = np.load('model/alex_mean.npy').swapaxes(0,1).swapaxes(1,2).astype(np.float32)
    alex_mean_images = tf.constant( alex_mean_npy )
    alex_images = alex_images[:,] - alex_mean_images
    alex_images = tf.slice(alex_images,[0,14,14,0],[FLAGS.batch_size,227,227,3])
    _,logits_alex = self.network_fn_alex(alex_images)
    logits = [logits_vgg16,logits_res,logits_incepv3,logits_incepv4,logits_incep_res,logits_alex]
    ensemble_logits = tf.reduce_mean(tf.stack(logits),0)
    return ensemble_logits 
  def all_feats(self,x_input):
    if(self.build):
      tf.get_variable_scope().reuse_variables()
    else:
      self.build = True
    resized_images_vgg = tf.image.resize_images(x_input,[224,224]) - tf.constant([123.68,116.78,103.94])
    with slim.arg_scope(vgg.vgg_arg_scope()):
      _, end_points_vgg16 = self.network_fn_vgg16(resized_images_vgg,num_classes=self.num_classes,is_training=False)
    resized_images_res = (tf.image.resize_images(x_input,[224,224])/255.0-0.5)*2
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      _, end_points_res = self.network_fn_res(resized_images_res,num_classes=self.num_classes+1,is_training=False)
    probs = []
    for layer in ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 
            'vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2', 
            'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2', 
            'vgg_16/conv3/conv3_3', 'vgg_16/conv4/conv4_1', 
            'vgg_16/conv4/conv4_2', 'vgg_16/conv4/conv4_3', 
            'vgg_16/conv5/conv5_1', 'vgg_16/conv5/conv5_2', 'vgg_16/conv5/conv5_3', 
            'vgg_16/fc6', 'vgg_16/fc7','vgg_16/fc8']: 
      output = end_points_vgg16[layer]
      probs.append(output)
    for layer in ['resnet_v2_152_1/block3/unit_23/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_24/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_25/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_26/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_27/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_28/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_29/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_31/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_32/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_33/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_34/bottleneck_v2', 
            'resnet_v2_152_1/block3/unit_36/bottleneck_v2', 
            'resnet_v2_152_1/block4/unit_3/bottleneck_v2']: 
      output = end_points_res[layer]
      probs.append(output)
    return probs

def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  src_image_height = 299
  src_image_width = 299
  eps = FLAGS.max_epsilon 
  pixel_scale = 255.0/(2*eps)
  nb_iter = 5 
  src_batch_shape = [FLAGS.batch_size, src_image_height, src_image_width, 3]
  batch_shape = [FLAGS.batch_size, src_image_height, src_image_width, 3]
  num_classes = 1000
  feat_scale = 0.05 
  conv_scale = 10.0
  pow_scale = 0.5

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    label_input = tf.placeholder(tf.float32, shape=[FLAGS.batch_size,num_classes])
    src_x_input = tf.placeholder(tf.float32, shape=src_batch_shape)
    ensemble_model = EnsembleModel(num_classes)
    
    featmd_params = {'eps': eps,
              'nb_iter':nb_iter,
              'eps_iter':eps*2/nb_iter}
    featmd = OwnMethod(ensemble_model)
    
    grads = featmd.generate(x_input,feat_scale,conv_scale,pow_scale, **featmd_params)
    ensemble_grads = tf.sign(grads) * eps

    # Run computation
    variables_to_restore = slim.get_variables_to_restore()
    gen_adv_variables = []
    resnet_gen_adv_variables = []
    inceptionv3_gen_adv_variables = []
    inceptionv4_gen_adv_variables = []
    predict_variables = []
    for var in variables_to_restore:
      if(var.name.startswith('InceptionResnetV2')):
        predict_variables.append(var)
      elif(var.name.startswith('InceptionV3')):
        inceptionv3_gen_adv_variables.append(var)
      elif(var.name.startswith('InceptionV4')):
        inceptionv4_gen_adv_variables.append(var)
      elif(var.name.startswith('resnet_v2')):
        resnet_gen_adv_variables.append(var)
      elif(var.name.startswith('vgg_16')):
        gen_adv_variables.append(var)
    gen_adv_saver = tf.train.Saver(gen_adv_variables)
    resnet_gen_adv_saver = tf.train.Saver(resnet_gen_adv_variables)
    inceptionv3_gen_adv_saver = tf.train.Saver(inceptionv3_gen_adv_variables)
    inceptionv4_gen_adv_saver = tf.train.Saver(inceptionv4_gen_adv_variables)
    predict_saver = tf.train.Saver(predict_variables)
    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    step_index = 0
    with tf.Session(config=_config) as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      gen_adv_saver.restore(sess,'model/vgg_16.ckpt')
      resnet_gen_adv_saver.restore(sess,'model/resnet_v2_152.ckpt')
      predict_saver.restore(sess,'./model/ens_adv_inception_resnet_v2.ckpt')
      inceptionv3_gen_adv_saver.restore(sess,'model/inception_v3.ckpt')
      inceptionv4_gen_adv_saver.restore(sess,'model/inception_v4.ckpt')
      for filenames, src_images in load_images(FLAGS.input_dir,batch_shape,src_batch_shape):
        print ('step: %d' % step_index)
        inception_images = (src_images.astype(np.float)/255.0-0.5)*2;
        adv_grads = sess.run(ensemble_grads, feed_dict={x_input: src_images})
        adv_images = src_images + adv_grads
        adv_images = np.clip(adv_images,0,255)
        save_images(adv_images, filenames, FLAGS.output_dir)
        step_index += 1

if __name__ == '__main__':
  tf.app.run()
