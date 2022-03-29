import pickle as pkl
from results.metrics import * 
from data.datasets import get_dataset, DATASETS_CONFIG

import sys 

file = sys.argv[1]


with open(file, 'rb') as f:
	history, classes, config = pkl.load(f)


dataset = get_dataset(config)

proportions_of_added_pixels_ = proportions_of_added_pixels(dataset, history, config)

import pdb 
pdb.set_trace()