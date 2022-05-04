import numpy as np
import pandas as pd 
import dash_table


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

def build_dash_table(dict_, transpose=False, round=None):
    df = pd.DataFrame.from_dict(dict_)
    if transpose:
        df = df.T
    if round is not None:
        df.round(round)
    df = dash_table.DataTable(
            id='train_metrics',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'))

    return df

def spectra_bbm(spectra, mask_bands):
    """
    Args:
        - spectra: npy array, HS cube
        - mask_bands: npy boolean array, masked bands
    Output:
        HS cube with NaN at masked band locations
    """
    if mask_bands is not None:
        mask_bands = np.array(mask_bands).astype(bool)
        res = np.zeros((spectra.shape[0],len(mask_bands)))
        res[:, mask_bands] = spectra
        res[:, mask_bands==False] = np.nan
        return res
    else:
        return spectra
