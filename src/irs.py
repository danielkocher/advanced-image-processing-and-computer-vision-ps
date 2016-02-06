################################################################################
# IRS - Image Recognition System
# Advanced Image Processing & Computer Vision class at University of Salzburg.
# 
# Author: Daniel Kocher
################################################################################

################################################################################
# Main file (entry point of the project).
################################################################################

import sys
import os
import numpy
#import sklearn
import cv2

import settings.settings as settings

def main():
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

def read_file(path):
  content = []
  with open(path, 'r') as f:
    for line in f:
      content.append(line.strip())
  return content

def split_entries(l, delimiter):
  l_new = []
  for entry in l:
    l_new.append([float(i) for i in entry.split()])
  return l_new

if __name__ == "__main__":
  main()
