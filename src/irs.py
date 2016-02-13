################################################################################
# IRS - Image Recognition System
# Advanced Image Processing & Computer Vision class at University of Salzburg.
# 
# Author: Daniel Kocher
################################################################################

################################################################################
# Main file (entry point of the project).
################################################################################

from collections import defaultdict

import settings.settings as settings
import scaler.scaler as scaler
import kmeanspp.kmeanspp as kmeanspp
import feature_extraction.feature_extraction as fe
import bow.bow as bow
import scene_recognition.scene_recognition as scene_rec

#
def main ():
  computed_feature_vectors = {} # used to avoid recomputation of SIFT features

  try:
    settings.init()
    settings.check_settings()
    settings.print_settings()
  except NameError as ne:
    print('NameError: {0}'.format(ne))

  attributes = read_file(settings.filepaths['attributes'])
  images = read_file(settings.filepaths['images'])
  votes_tmp = read_file(settings.filepaths['votes'])
  votes = split_entries(votes_tmp, ' ')

  # check if min-max-scaler was already instantiated
  # if so, open and use it
  # otherwise, geneate min-max-scaler pickle file and use it
  try:
    used_scaler = scaler.open_if_exists()
    print('[IRS] Using existing scaler pickle file')
  except IOError as ioe:
    print('[IRS] Scaler pickle file does not exist')
    print('[IRS] Hence it will now be created (may take some time)')
    used_scaler = scaler.create(images, computed_feature_vectors)

  # check if k-means clustering was already done
  # if so, open and use it
  # otherwise, generate k-means clustering pickle file and use it
  try:
    used_kmeanspp = kmeanspp.open_if_exists()
    print('[IRS] Using existing k-means++ pickle file')
  except IOError as ioe:
    print('[IRS] k-means++ pickle file does not exist')
    print('[IRS] Hence this will now be created (may take some time)')
    used_kmeanspp = kmeanspp.create(images, computed_feature_vectors, used_scaler)

  #
  ai_dict, aic_dict = map_images_to_attributes(attributes, images, votes)

  # generate splits and train classifiers
  classifiers = bow.learn_and_evaluate(attributes, ai_dict, aic_dict,
    'symmetric', './', computed_feature_vectors, used_scaler, used_kmeanspp
  )
  print('Trained classifiers for {} attributes (10 each; total: {})'.format(
    len(classifiers), get_total_classifier_count(classifiers)
  ))

  # recognize scenes
  scene_rec.learn_and_evaluate(used_scaler, used_kmeanspp, classifiers)


#
def get_total_classifier_count (classifiers):
  total_classifier_count = 0
  for attribute, classifier_list in classifiers.iteritems():
    total_classifier_count += len(classifier_list)
  return total_classifier_count

# Reads a given file.
# Returns a list of the read lines.
def read_file (path):
  content = []
  with open(path, 'r') as f:
    for line in f:
      content.append(line.strip())
  return content

# Splits each element of a given list by a given delimiter and converts each
# entry to a float.
# Returns a list of lists with float entries (representing the votes)
def split_entries (l, delimiter):
  l_new = []
  for element in l:
    l_new.append([float(i) for i in element.split()])
  return l_new

# Maps images to attributes given the attributes list, images list and votes list.
# Assumes corresponding indices between the three lists.
# Returns two dictionaries:
#  (1) maps attributes to images where the attribute is present
#  (2) the complement of the first one (w.r.t. the all attributes/images)
def map_images_to_attributes (attributes, images, votes):
  ai_dict = defaultdict(list)
  aic_dict = defaultdict(list)
  vote_bound = float(2)/float(3)

  for image_index, image in enumerate(images):
    for attribute_index, attribute in enumerate(attributes):
      vote = votes[image_index][attribute_index]
      if vote < vote_bound:
        aic_dict[attribute].append(image)
      else:
        ai_dict[attribute].append(image)
     
  return [ai_dict, aic_dict]

if __name__ == "__main__":
  main()
