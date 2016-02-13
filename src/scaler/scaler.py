################################################################################
# Implements a Min-Max-Scaler based upon SIFT feature extraction. 
#
# Author: Daniel Kocher
################################################################################

from __future__ import print_function
import os
import sys
import pickle as pe
import numpy as np
from sklearn import preprocessing
import random

import settings.settings as settings
import feature_extraction.feature_extraction as fe

#
def open_if_exists ():
  try:
    return pe.load(open(settings.filepaths['scaler'], 'rb'))
  except IOError, e:
    raise

#
def create (images, computed_feature_vectors):
  feature_vectors = []
  step_mod = len(images) * 0.05 # print progress in steps of 5 percent
  done = 0
  print('[SCALER] SIFT feature extraction:\n[SCALER] 0%', end = '')
  sys.stdout.flush()
  for image in images:
    if image in computed_feature_vectors:
      descriptors = computed_feature_vectors[image]
    else:
      try:
        descriptors = fe.extract(settings.images_dir + image, settings.scale_size)
      except Exception as e:
        print('Error while computing SIFT for image {}.'.format(image))
        continue
      computed_feature_vectors[image] = descriptors

    fe.unroll_and_append_descriptors(descriptors, feature_vectors)
    done += 1
    if done % step_mod == 0:
      print(' {}%'.format((100 * done) / len(images)), end = '')
      sys.stdout.flush()
  
  if done < len(images):
    print(' 100%')
  print('[SCALER] Number of \'SIFTed\' images: {}'.format(done))
  print('[SCALER] Total number of SIFT descriptors: {}'
    .format(len(feature_vectors))
  )
  sys.stdout.flush()

  # calibrate min-max-scaler
  min_max_scaler = preprocessing.MinMaxScaler()
  # 10 SIFT descriptors per image (approximately)
  min_max_scaler = min_max_scaler.fit(
    random.sample(feature_vectors, int(10 * done))
  )

  # dump min-max-scaler for further usage
  with open(settings.filepaths['scaler'], 'wb') as f:
    pe.dump(min_max_scaler, f)

  return min_max_scaler
