import numpy as np 
import matplotlib.pyplot as plt 
from results.utils import make_gt

def proportions_of_added_pixels(dataset, history, config):
    """
    Returns a (n_classes x n_steps) npy array of proportions of added pixels per class
    """
    labels = history['labels']
    n_steps = config['steps']
    n_px = config['n_px']
    n_px_per_class = dataset.n_px_per_class
    classes = np.arange(0, dataset.train_gt().max()+1)
    added_classes = np.zeros((len(classes)-1, n_steps))
    assert n_px == len(labels)//n_steps
    for step in range(n_steps):
        added = labels[step*n_px:(step+1)*n_px]
        for class_id in np.unique(added):
            added_classes[class_id-1, step] = sum(added == class_id)
    added_classes = np.cumsum(added_classes, axis=1)
    return  added_classes / n_px_per_class.reshape(-1,1)


def accuracy_metrics(model, dataset, history, config):
    accuracy_metrics = {
        'OA': np.zeros(config['steps']),
        'mIoU': np.zeros(config['steps']),
        'cm': np.zeros((config['steps'], dataset.n_classes, dataset.n_classes))
        }

    for step in range(1, config['steps']+1):
        dataset.train_gt.gt, dataset.pool.gt = make_gt(dataset, history, config, step)
        model.train(dataset, config)
        metrics = model.evaluate(dataset, config)
        for metric in accuracy_metrics:
            accuracy_metrics[metric][step-1] = metrics[metric]

    return accuracy_metrics

