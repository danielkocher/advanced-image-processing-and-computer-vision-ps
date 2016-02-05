################################################################################
# IRS - Image Recognition System
# Advanced Image Processing & Computer Vision class at University of Salzburg.
# 
# Author: Daniel Kocher
#
################################################################################

################################################################################
# Main file (entry point of the project).
################################################################################

import sys
import os
import numpy
import sklearn

import settings.settings as settings

def main():
  try:
    settings.init()
    settings.check_settings()
    settings.print_settings()
  except NameError as ne:
    print('NameError: {0}'.format(ne))

if __name__ == "__main__":
  main()
