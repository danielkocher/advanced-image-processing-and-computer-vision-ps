################################################################################
# Implements basic k-means++ clustering based upon SIFT feature extraction. 
#
# Author: Daniel Kocher
################################################################################

from __future__ import print_function
import os
import sys
import pickle as pe
import numpy as np
from sklearn import preprocessing, cluster
import random

import settings.settings as settings
import feature_extraction.feature_extraction as fe

#
def open_if_exists ():
  try:
    return pe.load(open(settings.filepaths['k-means++'], 'rb'))
  except IOError, e:
    raise

#
def create (images, computed_feature_vectors, scaler):
  feature_vectors = []
  step_mod = len(images) * 0.05 # print progress in steps of 5 percent
  done = 0
  print('[K-MEANS++] SIFT feature extraction:\n[K-MEANS++] 0%', end = '')
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
  print('[K-MEANS++] Number of \'SIFTed\' images: {}'.format(done))
  print('[K-MEANS++] Total number of sift descriptors: {}'
    .format(len(feature_vectors))
  )
  sys.stdout.flush()

  # do k-means clustering (512 clusters, k-means++, 4 cores, show information)
  kmeans = cluster.KMeans(n_clusters = settings.class_count, init = 'k-means++',
    n_jobs = 4, verbose = 1
  )
  # 10 SIFT descriptors per images (approximately)
  kmeans.fit(scaler.transform(
    random.sample(feature_vectors, int(10 * done))
  ))

  # dump k-means clustering for further usage
  with open(settings.filepaths['k-means++'], 'wb') as f:
    pe.dump(kmeans, f)

  return kmeans
