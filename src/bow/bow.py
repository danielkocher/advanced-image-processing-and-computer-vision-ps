################################################################################
# Essentially implements a BoW representation.
#
# Author: Daniel Kocher
################################################################################

from __future__ import print_function
import sys
from random import choice, shuffle
import cv2
from sklearn import svm, linear_model
import pickle as pe
import numpy as np

import settings.settings as settings
import feature_extraction.feature_extraction as fe

# Select 'size' random images from a list of candidates and store feature vectors
# to avoid recomputation.
def select_random_images (size, candidates, computed_feature_vectors,
  constraints = None):
  selected_candidates = []
  i = 0
  while i < size:
    image = settings.images_dir + choice(candidates)

    # avoid duplicates
    if image in selected_candidates:
      continue;

    # make sure the constraints are not in the candidates list
    if constraints != None and image in constraints:
      continue;

    if image in computed_feature_vectors:
      descriptors = computed_feature_vectors[image]
    else:
      try:
        descriptors = fe.extract(image, settings.scale_size)
        computed_feature_vectors[image] = descriptors
      except Exception as e:
        print('Error while computing SIFT for image {}'.format(image))
        continue

    selected_candidates.append(image)
    i += 1
  return selected_candidates

#
def generate_splits (attribute, i_list, ic_list, mode, path,
  computed_feature_vectors, scaler):
  if not mode in settings.train_test_sizes:
    print('Mode ' + mode +
      ' not found (must be either \'symmetric\' or \'asymmetric\')'
    )
    return
  
  train_set = { 1 : [], 0 : [] }
  test_set = { 1 : [], 0 : [] }
  # 1 = positive samples, 0 = negative samples
  # negative train samples
  train_set[0] = select_random_images(
    settings.train_test_sizes[mode]['train'][0], ic_list,
    computed_feature_vectors
  )
  # positive train samples
  train_set[1] = select_random_images(
    settings.train_test_sizes[mode]['train'][1], i_list,
    computed_feature_vectors
  )
  # negative test samples
  test_set[0] = select_random_images(
    settings.train_test_sizes[mode]['test'][0], ic_list,
    computed_feature_vectors, train_set[0]
  )
  # positive test samples
  test_set[1] = select_random_images(
    settings.train_test_sizes[mode]['test'][1], i_list,
    computed_feature_vectors, train_set[1]
  )

  #shuffle(train_set[1])
  #shuffle(train_set[0])
  #shuffle(test_set[1])
  #shuffle(test_set[0])
  
  feature_vectors = {
    'train' : { 1 : [], 0 : [] },
    'test' : { 1 : [], 0 : [] }
  }
  for i in range(0, 2):
    # train
    for train_image in train_set[i]:
      #if train_image in computed_feature_vectors:
      descriptors = computed_feature_vectors[train_image]
      feature_vectors['train'][i].append(
        scaler.transform(descriptors)
      )
    # test
    for test_image in test_set[i]:
      #if test_image in computed_feature_vectors:
      descriptors = computed_feature_vectors[test_image]
      feature_vectors['test'][i].append(
        scaler.transform(descriptors)
      )
 
  return feature_vectors

#
def learn_and_evaluate (attributes, ai_dict, aic_dict, mode, path,
  computed_feature_vectors, scaler, kmeanspp):
  aps_stds = []
  total_number_of_positives = settings.train_test_sizes[mode]['train'][0] + settings.train_test_sizes[mode]['test'][0]
  classifiers = {}
  for attribute in attributes:
    # for the asymmetric mode: only evaluate attributes, for which enough positive
    # images are present (=> 87 attributes in total according to reference splits)
    if len(ai_dict[attribute]) < total_number_of_positives:
      continue

    print('[BoW] Attribute: {}'.format(attribute), end = '')
    sys.stdout.flush()
    
    # if 'splits_per_attribute' classifiers already exist, use them
    classifier_prefix = settings.classifiers_dir + attribute + '-classifier-'
    classifiers[attribute] = []
    for split in range(settings.splits_per_attribute):
      try:
        classifiers[attribute].append(
          pe.load(open(classifier_prefix + str(split) + '.p', 'rb'))
        )
      except IOError as ioe:
        break;

    if len(classifiers[attribute]) == settings.splits_per_attribute:
      print(' (loaded from files)')
      continue

    scores = []
    for split in range(settings.splits_per_attribute):
      print(' {}'.format(split + 1), end = '')
      sys.stdout.flush()

      feature_vectors = generate_splits(attribute, ai_dict[attribute],
        aic_dict[attribute], mode, path, computed_feature_vectors, scaler
      )

      # generate histograms from k-means++ predictions
      train_histograms = []
      test_histograms = []
      for i in range(0, 2):
        for train_feature_vector in feature_vectors['train'][i]:
          kmeanspp_predicted = kmeanspp.predict(train_feature_vector)
          train_histograms.append(generate_histogram(kmeanspp_predicted))
        for test_feature_vector in feature_vectors['test'][i]:
          kmeanspp_predicted = kmeanspp.predict(test_feature_vector)
          test_histograms.append(generate_histogram(kmeanspp_predicted))

      # train classifier (from liblinear as suggested in the PS)
      classifier = svm.LinearSVC()
      #classifier = linear_model.LogisticRegression()
      classifier.fit(train_histograms, settings.train_test_labels[mode]['train'])
      # compute score
      score = classifier.score(
        test_histograms, settings.train_test_labels[mode]['test']
      )
     
      # store classifier and score
      classifiers[attribute].append(classifier)
      scores.append(score)
    
      with open(classifier_prefix + str(split) + '.p', 'wb') as f:
        pe.dump(classifier, f)
    print(' (new computation)')

    # compute average precision and standard deviation
    ap = sum(scores)/len(scores)
    std = np.std(scores)
    print('[BoW] AP: {}, STD: {}'.format(ap, std))
    aps_stds.append([attribute, ap, std])

  # save average precision and standard deviation for illustration
  aps_stds.sort(key = lambda tup: tup[1], reverse = True)
  with open(settings.src_dir + 'scores', 'a') as f:
    for i, entry in enumerate(aps_stds):
      f.write(str(i) + ' ' + str(entry[0]) + ' ' + str(entry[1]) + ' ' +
        str(entry[2]) + '\n'
      )
  return classifiers


# Generates histogram
#
# Note: This method has to make sure that the histograms are of equal
# dimensionality in order to be able to use them in the SVM (fit)!
def generate_histogram (predicted):
  histogram = [float(0) for i in range(settings.class_count)]
  for p in predicted:
    histogram[p] += float(1)
  return histogram
