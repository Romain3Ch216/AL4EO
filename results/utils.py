import numpy as np

def make_gt(dataset, history, config, step):
    n_px = config['n_px']
    coordinates, class_ids = history['coordinates'], history['labels']
    coordinates = np.array(coordinates)
    train_gt = np.copy(dataset.train_gt())
    pool = np.copy(dataset.pool())
    for i in range(step*n_px):
        train_gt[coordinates[i,0], coordinates[i,1]] = class_ids[i]
        pool[coordinates[i,0], coordinates[i,1]] = 0
    return train_gt, pool