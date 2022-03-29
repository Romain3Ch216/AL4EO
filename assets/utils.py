import pandas as pd
import dash_table
import numpy as np
import pdb

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
    mask_bands = np.array(mask_bands).astype(bool)
    res = np.zeros((spectra.shape[0],len(mask_bands)))
    res[:, mask_bands] = spectra
    res[:, mask_bands==False] = np.nan
    return res

def make_gt(added, dataset, step, n_px):
    coordinates, class_ids = added['coordinates'], added['labels']
    coordinates = np.array(coordinates)
    train_gt = np.copy(dataset.train_gt())
    pool = np.copy(dataset.pool())
    for i in range(step*n_px):
        train_gt[coordinates[i,0], coordinates[i,1]] = class_ids[i]
        pool[coordinates[i,0], coordinates[i,1]] = 0
    return train_gt, pool

def load_data(img, train_gt, val_gt, kwargs):
    train_dataset = HyperX(img, train_gt, **kwargs)
    train_loader  = data.DataLoader(train_dataset,shuffle=True,
                              batch_size=kwargs['batch_size'])

    val_dataset = HyperX(img, val_gt, **kwargs)
    val_loader  = data.DataLoader(val_dataset,shuffle=True,
                              batch_size=kwargs['batch_size'])
    return train_loader, val_loader

def get_indices_as_array(img_shape: tuple) -> np.ndarray:
    indices = np.zeros(img_shape)
    indices = np.where(indices == 0)
    indices = np.array(indices)
    return indices.T

def confusion_matrix_analysis(mat):
    """
    This method computes all the performance metrics from the confusion matrix. In addition to overall accuracy, the
    precision, recall, f-score and IoU for each class is computed.
    The class-wise metrics are averaged to provide overall indicators in two ways (MICRO and MACRO average)
    Args:
        mat (array): confusion matrix

    Returns:
        per_class (dict) : per class metrics
        overall (dict): overall metrics

    """
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = np.sum(mat[j, j])
        fp = np.sum(mat[:, j]) - tp
        fn = np.sum(mat[j, :]) - tp

        d['IoU'] = round((100 * tp / (tp + fp + fn + 1e-20)), 1)
        d['Precision'] = round((100 * tp / (tp + fp + 1e-20)), 1)
        d['Recall'] = round((100 * tp / (tp + fn + 1e-20)), 1)
        d['F1-score'] = round((100 * 2 * tp / (2 * tp + fp + fn + 1e-20)), 1)

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    overall['micro_IoU'] = round((100 * TP / (TP + FP + FN)), 1)
    overall['micro_Precision'] = round((100 * TP / (TP + FP)), 1)
    overall['micro_Recall'] = round((100 * TP / (TP + FN)), 1)
    overall['micro_F1-score'] = round((100 * 2 * TP / (2 * TP + FP + FN)), 1)

    macro = pd.DataFrame(per_class).transpose().mean()
    overall['MACRO_IoU'] = round(macro.loc['IoU'], 1)
    overall['MACRO_Precision'] = round(macro.loc['Precision'], 1)
    overall['MACRO_Recall'] = round(macro.loc['Recall'], 1)
    overall['MACRO_F1-score'] = round(macro.loc['F1-score'], 1)

    overall['Accuracy'] = round((100 * np.sum(np.diag(mat)) / np.sum(mat)), 1)

    return per_class, overall
