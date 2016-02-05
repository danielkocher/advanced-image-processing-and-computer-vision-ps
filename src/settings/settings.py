################################################################################
# Config file providing some global used variables (e.g. paths)
#
# Author: Daniel Kocher
################################################################################

import os

# def of dirs and files
def init():
  global data_dir
  global images_dir
  global filepaths

  data_dir = '/home/seniix/github/advanced-image-processing-and-computer-vision-ps/data/'
  images_dir = data_dir + 'images/'

  filepaths = {
    'attributes' : data_dir + 'SUN_attributes.txt',
    'votes' : data_dir + 'SUN_attribute_votes.txt',
    'images' : data_dir + 'SUN_images.txt'
  }
  
# check stati of dirs and files
def check_settings():
  if not os.path.isdir(data_dir):
    raise NameError('Data directory \"' + data_dir + '\" does not exist.')

  if not os.path.isdir(images_dir): 
    raise NameError('Images directory \"' + images_dir +
      '\" does not exist.'
    )

  if not os.path.isfile(filepaths['attributes']):
    raise NameError('Attributes file \"' + filepaths['attributes'] +
      '\" does not exists.'
    )

  if not os.path.isfile(filepaths['votes']):
    raise NameError('Attribute votes file \"' + filepaths['votes'] +
      '\" does not exists.'
    )

  if not os.path.isfile(filepaths['images']):
    raise NameError('Images file \"' + filepaths['images'] +
      '\" does not exists.'
    )

# print dir and file settings
def print_settings():
  print('Settings:')
  print('  * data directory:       ' + data_dir)
  print('  * images directory:     ' + images_dir)
  print('  * attributes file:      ' + filepaths['attributes'])
  print('  * attribute votes file: ' + filepaths['votes'])
  print('  * images file:          ' + filepaths['images'])

