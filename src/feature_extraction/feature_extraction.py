################################################################################
# Provides methods to do/check for feature extraction.
#
# Author: Daniel Kocher
################################################################################

import os
import errno
import numpy as np
import cv2
from sklearn.decomposition import PCA

import settings.settings as settings

#
def create_feature_extraction_dir ():
  try:  
    os.makedirs(settings.feature_extraction_dir)
  except OSError as ose:
    if ose.errno != errno.EEXIST:
      raise

#
def create_sift_file_if_not_exists (attribute):
  try:
    create_feature_extraction_dir()
    filename = settings.feature_extraction_dir + attribute + '.sift'
    file_handle = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    return filename
  except OSError as ose:
    if ose.errno != errno.EEXIST:
      raise

#
def extract (image, scale_size):
  return dense_sift(image, scale_size)

#
def sift (image, scale_size):
  preprocessed_image = preprocess_image(image, scale_size)

  sift = cv2.xfeatures2d.SIFT_create()
  (keypoints, descriptors) = sift.detectAndCompute(preprocessed_image, None)
  
  return descriptors

#
def dense_sift (image, scale_size):
  preprocessed_image = preprocess_image(image, scale_size)
  
  # DenseFeatureDetector is gone in version >= 3.0
  # Reference: http://answers.opencv.org/question/73165/compute-dense-sift-features-in-opencv-30/
  
  # Hence, I create my own grid of keypoints
  step = 16 # 16 pixels spacing between keypoints; 256x256 image => = 16x16 grid 
  keypoints = []
  for i in range(step/2, preprocessed_image.shape[0], step): # rows
    for j in range(step/2, preprocessed_image.shape[1], step): # columns
      keypoints.append(cv2.KeyPoint(float(j), float(i), _size = float(step+4)))

  sift = cv2.xfeatures2d.SIFT_create()
  descriptors = []
  keypoints, descriptors = sift.compute(preprocessed_image, keypoints)

  drawn_kps = cv2.drawKeypoints(preprocessed_image, keypoints, preprocessed_image, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.imwrite('/home/seniix/Desktop/kps.jpg', drawn_kps)
  return descriptors

#
def preprocess_image (image, scale_size):
  gs_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
  return cv2.resize(gs_image, (scale_size, scale_size))

# 
def unroll_and_append_descriptors (descriptors, feature_vectors):
  for descriptor in descriptors:
    feature_vectors.append(descriptor)
