import os
import sys

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
pparent = os.path.dirname(parent)
  
# adding the parent directory to 
# the sys.path.
sys.path.append(pparent)

# import from 'fyp-ELCO/phase2/data_filepath'
from data_filepath import *

