"""
Script that performs active learning over a benchmark dataset
"""

import argparse
from learning.session import ActiveLearningFramework
from data.datasets import get_dataset, DATASETS_CONFIG
import pprint
import pdb
from assets.utils import *
from learning.query import load_query
import os
import errno
from path import get_path
import torch
import ast

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, help="Dataset to use.")
parser.add_argument('--model', type=str, default='cnn',
                    help="Model to train. Available:\n"
                    "Very Shallow FNN, "
                    "Deep Spectral CNN, "
                    "hu (1D CNN)")
parser.add_argument('--query', type=str, default='breaking_ties', help='Query system to use')
parser.add_argument('--n_px', type=int, default=10, help='Number of unlabled pixels to select at each iteration')
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--device', type=str, default='cpu',
                    help="Specify cpu or gpu")
parser.add_argument('--restore', type=str, default=False,
                    help="Weights to use for initialization, e.g. a checkpoint")
parser.add_argument('--res_dir', type=str, default='',
                    help="Results folder")
parser.add_argument('--toy_example', action="store_true",
                    help="Generate a toy example dataset")
parser.add_argument('--dev', action="store_true",
                    help="If in development, does not save")
parser.add_argument('--steps', type=int,
                    help="Number of AL steps")
parser.add_argument('--run', type=str,
                    help="name of run")
parser.add_argument('--timestamp', type=str,
                    help="timestamp")
parser.add_argument('--remove', type=str, default="[]",
                    help="Classes to remove at initial gt")
parser.add_argument('--opening', action='store_true', help='apply morphological profiles.')

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epochs', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--dropout', type=float, default=0., help="Dropout probability")
group_train.add_argument('--patch_size', type=int,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--weight_decay', type=float, default=0,
                    help="weight_decay, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
group_train.add_argument('--display_step', type=int, default=50,
                     help='Interval in batches between display of training metrics')
group_train.add_argument('--penalty', type=float, default=0,
                     help='Coefficient for penalty')
group_train.add_argument('--classes', type=str, help='Classes id to keep', default='all')
group_train.add_argument('--num_samples', type=int, help='Number of samples drawn from bayesian model', default=10)
group_train.add_argument('--num_draw', type=int, help='Number of samples to estimate the joint entropy of batch_bald', default=None)
group_train.add_argument('--Q', type=int, help='LAL hyperparams', default=None)
group_train.add_argument('--M', type=int, help='LAL hyperparams', default=None)
group_train.add_argument('--tau', type=str, help='LAL hyperparams', default=None)
group_train.add_argument('--outlier_prop', type=float, help='Coreset outliers budget', default=None)
group_train.add_argument('--beta', type=float, default=None, help='Hierarchical hyperparam')
group_train.add_argument('--coordinates', action="store_true")
group_train.add_argument('--subsample', type=float, default=1)
group_train.add_argument('--cluster', action="store_true")

args = parser.parse_args()

config = parser.parse_args()
config = vars(config)

if config['tau']:
    config['tau'] = ast.literal_eval(config['tau'])

if config['subsample'] == 1:
    config['subsample'] = False

config['remove'] = ast.literal_eval(config['remove'])
config = {k: v for k, v in config.items() if v is not None}

dataset = get_dataset(config)

save = not args.dev
run = args.run

if config['toy_example']:
    dataset.toy_example()

config['n_classes'] = dataset.n_classes
config['proportions'] = dataset.proportions
config['classes'] = np.unique(dataset.train_gt())[1:]
config['n_bands']   = dataset.n_bands
config['ignored_labels'] = dataset.ignored_labels
# config['n_pool'] = dataset.pool.size
# config['n_train'] = dataset.train_gt.size
config['img_shape'] = dataset.img.shape[:-1]
config['res_dir'] = '{}/Results/ActiveLearning/'.format(get_path()) + config['dataset'] + '/' + run + '/' + config['query']
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['benchmark'] = False if args.cluster else True

if config['benchmark']:
    print('No segmentation...')
else:
    print('Segmentation...')
    dataset.clustering()
    
try:
    os.makedirs(config['res_dir'], exist_ok=True)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

model, query, config = load_query(config, dataset)
AL = ActiveLearningFramework(dataset, model, query, config)
AL.timestamp = args.timestamp

import time

for step in range(args.steps):
    t = time.time()
    print('==== STEP {} ===='.format(step))
    AL.step()
    AL.oracle()
    f = open(config['res_dir'] + '/log.txt', 'a')
    f.writelines(['Step {}: {}min \n'.format(step, (time.time() - t)/60)])
    f.close()

if save:
    AL.save()
