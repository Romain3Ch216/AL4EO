import pickle as pkl
from results.metrics import * 
from data.datasets import get_dataset, DATASETS_CONFIG
from learning.models import SVM
import sys 
import os 

file = sys.argv[1]

with open(file, 'rb') as f:
	history, classes, config = pkl.load(f)

dataset = get_dataset(config)
model = SVM()

proportions_of_added_pixels_ = proportions_of_added_pixels(dataset, history, config)
accuracy_metrics_ = accuracy_metrics(model, dataset, history, config)

pkl.dump((proportions_of_added_pixels_, accuracy_metrics_),\
          open(os.path.join(config['res_dir'], 'results_{}.pkl'.format(config['timestamp'])), 'wb'))
