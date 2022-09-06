"""
Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Véronique Achard, Laurent Risser, Béatrice Berthelot, Xavier Briottet.
Script that performs active learning over a benchmark dataset
"""

import argparse
from learning.session import ActiveLearningFramework
from data.datasets import get_dataset, DATASETS_CONFIG
import pprint
import pdb
from learning.query import load_query
import os
import errno
from path import get_path
import torch
import numpy as np
import ast

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, help="Dataset to use.")
parser.add_argument('--query', type=str, default='breaking_ties', help='Query system to use')
parser.add_argument('--n_px', type=int, default=10, help='Number of unlabled pixels to select at each iteration')
parser.add_argument('--steps', type=int, help="Number of AL steps")
parser.add_argument('--run', type=str, help="name of run")
parser.add_argument('--device', type=str, default='cpu', help="Specify cpu or gpu")
parser.add_argument('--dev', action="store_true", help="If in development, does not save history")
parser.add_argument('--timestamp', type=str, help="timestamp")
parser.add_argument('--remove', type=str, default="[]", help="Classes to remove at start")
parser.add_argument('--edge_detection', type=float, default=0, help='Apply a Sobel filter to detect and remove edges.')
parser.add_argument('--subsample', type=float, default=1, help='Subsample ratio to apply on the pool')
parser.add_argument('--superpixels', action="store_true", help='If True, apply segmentation to work on superpixels')
parser.add_argument('--restore', type=str, help='Path of history archive to restore')
parser.add_argument('--n_segments', type=int, default=5000, help='Number of superpixels for SLIC if superpixels is True')
parser.add_argument('--compactness', type=int, default=10, help='Compactness param for SLIC if superpixels is True')
parser.add_argument('--n_random', type=int, default=11, help='Number of pixels to draw within a superpixel')
parser.add_argument('--op', action='store_true', help='Only perform one step without the automatic oracle if True')


# Query options
query_options = parser.add_argument_group('Query')
query_options.add_argument('--num_draw', type=int, help='Number of samples to estimate the joint entropy of batch_bald', default=None)
query_options.add_argument('--Q', type=int, help='LAL hyperparams', default=None)
query_options.add_argument('--M', type=int, help='LAL hyperparams', default=None)
query_options.add_argument('--tau', type=str, help='LAL hyperparams', default=None)
query_options.add_argument('--outlier_prop', type=float, help='Coreset outliers budget', default=None)
query_options.add_argument('--beta', type=float, default=None, help='Hierarchical hyperparam')
query_options.add_argument('--cost_matrix', type=str, default=None, help='Path to cost of confusion matrix')

# Training options
training_options = parser.add_argument_group('Training')
training_options.add_argument('--epochs', type=int, help="Training epochs")
training_options.add_argument('--dropout', type=float, default=0., help="Dropout probability")
training_options.add_argument('--lr', type=float, help="Learning rate, set by the model if not specified.")
training_options.add_argument('--weight_decay', type=float, default=0, help="weight_decay, set by the model if not specified.")
training_options.add_argument('--batch_size', type=int, help="Batch size")
training_options.add_argument('--num_samples', type=int, help='Number of samples drawn from bayesian model', default=10)


args = parser.parse_args()
config = parser.parse_args()
config = vars(config)

if config['tau']:
    config['tau'] = ast.literal_eval(config['tau'])

if config['subsample'] == 1:
    config['subsample'] = False

assert (not config['subsample']) or (config['subsample'] and (config['query'] in ['coreset', 'hierarchical'])), "Subsampling is only available for Hierarchical and Coreset."

config['remove'] = ast.literal_eval(config['remove'])

config = {k: v for k, v in config.items() if v is not None}

dataset = get_dataset(config)
config['n_classes'] = dataset.n_classes
config['proportions'] = dataset.proportions
config['classes'] = np.unique(dataset.train_gt())[1:]
config['n_bands']   = dataset.n_bands
config['ignored_labels'] = dataset.ignored_labels
config['img_shape'] = dataset.img.shape[:-1]
config['res_dir'] = '{}/Results/ActiveLearning/'.format(get_path()) + config['dataset'] + '/' + args.run + '/' + config['query']

if config['superpixels']:
    print("Segment image in superpixels...")
    dataset.segmentation_(args.n_segments, args.compactness)

if config['query'] == 'probabilistic_breaking_tie':
    if 'cost_matrix' not in config:
        raise Exception("You should specify a cost of confusion matrix to use Probabilistic Breaking Tie") 
    dataset.cost_matrix = np.load(config['cost_matrix'])

try:
    os.makedirs(config['res_dir'], exist_ok=True)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

model, query, config = load_query(config, dataset)
AL = ActiveLearningFramework(dataset, model, query, config)

if args.op:
    AL.step()
    if 'restore' not in config:
        AL.config['step'] = 1
    else:
        AL.config['step'] += 1

else:
    for step in range(args.steps):
        print(f'==== STEP {step} ====')
        AL.step()
        AL.oracle()

if not args.dev:
    AL.save()
