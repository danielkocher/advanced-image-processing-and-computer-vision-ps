################################################################################
# Config file providing some global used variables (e.g. paths)
#
# Author: Daniel Kocher
################################################################################

import os
import errno

# def of dirs, files and constants
def init ():
  global data_dir
  global src_dir
  global images_dir
  global scaler_dir
  global kmeanspp_dir
  global splits_dir
  global classifiers_dir
  global scenes_dir
  global filepaths
  global splits_per_attribute
  global train_test_sizes
  global train_test_labels
  global scale_size
  global class_count
  global scene_train_ratio

  data_dir = '/home/seniix/github/advanced-image-processing-and-computer-vision-ps/data/'
  src_dir = '/home/seniix/github/advanced-image-processing-and-computer-vision-ps/src/'
  images_dir = data_dir + 'images/'
  scaler_dir = src_dir + 'scaler/'
  kmeanspp_dir = src_dir + 'kmeanspp/'
  splits_dir = data_dir + 'splits/'
  classifiers_dir = src_dir + 'classifiers/'
  scenes_dir = data_dir + 'scenes/'

  filepaths = {
    'attributes' : data_dir + 'SUN_attributes.txt',
    'votes' : data_dir + 'SUN_attribute_votes.txt',
    'images' : data_dir + 'SUN_images.txt',
    'k-means++' : kmeanspp_dir + 'kmeanspp.p',
    'scaler' : scaler_dir + 'scaler.p'
  }
  
  # 1 = positive sample, 0 = negative sample
  train_test_sizes = {
    'asymmetric' : { 'train' : [ 150, 150 ] , 'test' : [ 50, 50 ] },
    #'asymmetric' : { 'train' : [ 6, 6 ], 'test' : [ 2, 2 ] },
    'symmetric' : { 'train' : [ 150, 150 ], 'test' : [ 150, 150 ] }
  }

  # initialize labels (1 = positive sample, 0 = negative sample)
  train_test_labels = {
    'asymmetric' : {
      'train' : [],
      'test' : []
    },
    'symmetric' : {
      'train' : [],
      'test' : []
    }
  }
  label = 1
  for i in range(0, 2):
    for split in ['asymmetric', 'symmetric']:
      for t in ['train', 'test']:
        train_test_labels[split][t].extend(
          [ label for j in range(0, train_test_sizes[split][t][i]) ]
        )
    label = 0

  splits_per_attribute = 10
  scale_size = 256
  class_count = 512
  scene_train_ratio = 80 # percent
  
# check stati of dirs and files
def check_settings ():
  # required
  if not os.path.isdir(data_dir):
    raise NameError('Data directory \"' + data_dir + '\" does not exist.')

  # required
  if not os.path.isdir(images_dir): 
    raise NameError('Images directory \"' + images_dir +
      '\" does not exist.'
    )
 
  # required
  if not os.path.isdir(scenes_dir):
    raise NameError('Scenes directory \"' + scenes_dir +
      '\" does not exist.'
    )
 
  # required
  if not os.path.isfile(filepaths['attributes']):
    raise NameError('Attributes file \"' + filepaths['attributes'] +
      '\" does not exists.'
    )

  # required
  if not os.path.isfile(filepaths['votes']):
    raise NameError('Attribute votes file \"' + filepaths['votes'] +
      '\" does not exists.'
    )

  # required
  if not os.path.isfile(filepaths['images']):
    raise NameError('Images file \"' + filepaths['images'] +
      '\" does not exists.'
    )
 
  # optional (directory is created if it does not exist)
  try:
    os.makedirs(scaler_dir)
  except OSError as ose:
    if ose.errno != errno.EEXIST:
      raise

  # optional (directory is created if it does not exist)
  try:
    os.makedirs(kmeanspp_dir)
  except OSError as ose:
    if ose.errno != errno.EEXIST:
      raise

  # optional (directory is created if it does not exist)
  try:
    os.makedirs(classifiers_dir)
  except OSError as ose:
    if ose.errno != errno.EEXIST:
      raise

# print dir and file settings
def print_settings ():
  print('[SETTINGS] Settings:')
  print('[SETTINGS]  * data directory:          {}'.format(data_dir))
  print('[SETTINGS]  * source directory:        {}'.format(src_dir))
  print('[SETTINGS]  * images directory:        {}'.format(images_dir))
  print('[SETTINGS]  * scaler directory:        {}'.format(scaler_dir))
  print('[SETTINGS]  * k-means++ directory:     {}'.format(kmeanspp_dir))
  print('[SETTINGS]  * splits directory:        {}'.format(splits_dir))
  print('[SETTINGS]  * classifiers directory:   {}'.format(classifiers_dir))
  print('[SETTINGS]  * attributes file:         {}'.format(filepaths['attributes']))
  print('[SETTINGS]  * attribute votes file:    {}'.format(filepaths['votes']))
  print('[SETTINGS]  * images file:             {}'.format(filepaths['images']))
  print('[SETTINGS]  * scaler file:             {}'.format(filepaths['scaler']))
  print('[SETTINGS]  * k-means++ file:          {}'.format(filepaths['k-means++']))
  print('[SETTINGS]  * asymmetric (train/test): {}/{}'.format(
    train_test_sizes['asymmetric']['train'][0] + train_test_sizes['asymmetric']['train'][1],
    train_test_sizes['asymmetric']['test'][0] + train_test_sizes['asymmetric']['test'][1],
  ))
  print('[SETTINGS]  * symmetric (train/test):  {}/{}'.format(
    train_test_sizes['symmetric']['train'][0] + train_test_sizes['symmetric']['train'][1],
    train_test_sizes['symmetric']['test'][0] + train_test_sizes['symmetric']['test'][1],
  ))
  print('[SETTINGS]  * splits per attribute:    {}'.format(splits_per_attribute))
  print('[SETTINGS]  * scaling to:              {}x{}'.format(
    scale_size, scale_size
  ))
  print('[SETTINGS]  * number of classes:       {}'.format(class_count))
