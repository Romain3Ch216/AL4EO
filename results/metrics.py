import numpy as np 
import matplotlib.pyplot as plt 

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



def overall_accuracy(dataset, history, config):