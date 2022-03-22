import torch

from extractor import SAVE_FILEPATH
import pprint 
dataset = torch.load(SAVE_FILEPATH)
pprint.pprint(dataset[:2])