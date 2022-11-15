from data.data import Dataset
from path import get_path

DATASETS_CONFIG = {
    'Demo': {
        'label_values': ['Unclassified', 'Asphalt', 'Vegetation', 'Water'],
        'sensor': 'FENIX',
        'ignored_labels': [0],
        'palette': None,
        'img_pth': '{}/Datasets/Demo/boulevard_monplaisir.bsq'.format(get_path()),
        'gt_pth': '{}/Datasets/Demo/asphalt.bsq'.format(get_path())
    },
  }


def get_dataset(config, datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        datasets: dataset configuration dictionary
    Returns:
        Dataset object
    """
    dataset_name = config['dataset']
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    return Dataset(config, **DATASETS_CONFIG[dataset_name])
