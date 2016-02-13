################################################################################
# Implementation of all methods concerning the scene recognition.
#
# Author: Daniel Kocher
################################################################################

import os
from random import choice
from sklearn.neighbors import KNeighborsClassifier

import settings.settings as settings
import feature_extraction.feature_extraction as fe
import bow.bow as bow

#
def learn_and_evaluate (scaler, kmeanspp, classifiers):
  scene_images = collect_scene_images(settings.scenes_dir)
  scene_splits = split_scene_images(scene_images, settings.scene_train_ratio)

  # train
  train_attribute_vectors, train_scenes = create_attribute_vectors(
    scene_splits['train'], scaler, kmeanspp, classifiers
  )
  print('Train attribute vector generation completed ({})'.format(
    len(train_scenes))
  )
  
  # train k-nearest neighbors classifier
  kNN_classifier = KNeighborsClassifier(n_neighbors = 3)
  kNN_classifier.fit(train_attribute_vectors, train_scenes)
  print('k-NN classifier training completed')

  # test
  test_attribute_vectors, test_scenes = create_attribute_vectors(
    scene_splits['test'], scaler, kmeanspp, classifiers
  )
  print('Test attribute vector generation completed ({})'.format(
    len(test_scenes))
  )

  # score/predict using the k-nearest neighbors classifier trained beforehand
  begin_index = 0
  total_scores = []
  for i in range(1, len(test_scenes)):
    if test_scenes[begin_index] == test_scenes[i]:
      continue
    else:
      total_scores.append(predict_scene(test_scenes, test_attribute_vectors,
        begin_index, i, kNN_classifier
      ))
      begin_index = i
  total_scores.append(predict_scene(test_scenes, test_attribute_vectors,
    begin_index, len(test_scenes), kNN_classifier
  ))
  print('Total scores: {}'.format(total_scores))

#
def predict_scene (test_scenes, test_attribute_vectors, begin_index, end_index,
  kNN_classifier):
  sub_scenes = test_scenes[begin_index:end_index]
  sub_attribute_vectors = test_attribute_vectors[begin_index:end_index]
  kNN_predictions = kNN_classifier.predict(sub_attribute_vectors)
  print('scene {} predictions: {}'.format(sub_scenes[0], kNN_predictions))
  kNN_score = kNN_classifier.score(sub_attribute_vectors, sub_scenes)
  print('scene {} score: {}'.format(sub_scenes[0], kNN_score))
  return [sub_scenes[0], kNN_score]


#
def create_attribute_vectors (scene_splits, scaler, kmeanspp, classifiers):
  scenes = []
  attribute_vectors = []
  for subdir, images in scene_splits.iteritems():
    # skip root directory
    if len(images) == 0:
      continue
    
    # extract scene name
    scene_name = subdir[subdir.rfind('/') + 1:]

    for image in images:
      descriptors = fe.extract(image, settings.scale_size)
      kmeanspp_predicted = kmeanspp.predict(
        scaler.transform(descriptors)
      )
      histogram = bow.generate_histogram(kmeanspp_predicted)

      # predict attributes of all present classifiers
      # (87 for asymmetric, 102 for symmetric)
      attribute_vector = []
      for attribute, classifier_list in classifiers.iteritems():
        attribute_vector.extend(classifier_list[5].predict(histogram))
      
      # store scene and attribute vector
      scenes.append(scene_name)
      attribute_vectors.append(attribute_vector)
  return [attribute_vectors, scenes]


#
def collect_scene_images (scenes_dir):
  scene_images = {} # maps images to dirs
  for root, dirs, files in os.walk(scenes_dir):
    for subdir in dirs:
      scene_images[os.path.join(root, subdir)] = []
    scene_images[root] = [os.path.join(root, f) for f in files]
  return scene_images

#
def split_scene_images (scene_images, train_ratio):
  train_images = {}
  test_images = {}
  for subdir, images in scene_images.iteritems():  
    scene_image_count = len(images)
    ratio_image_count = int(scene_image_count * 0.8)
    difference = scene_image_count - ratio_image_count

    # train (use about 80 percent of the current scene category)
    train_images[subdir] = []
    while len(train_images[subdir]) < ratio_image_count:
      random_image = choice(images)
      if random_image in train_images[subdir]:
        continue
      train_images[subdir].append(random_image)
    # test (add remaining)
    test_images[subdir] = []
    for image in images:
      if image in train_images[subdir]:
        continue
      test_images[subdir].append(image)
  
  return { 'train' : train_images, 'test' : test_images }
