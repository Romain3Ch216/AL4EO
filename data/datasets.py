from data.data import Dataset
from path import get_path
import pdb

DATASETS_CONFIG = {

    'Houston': {
        'label_values': ['unclassified', 'healthy_grass', 'stressed_grassed',
                         'artificial_turf', 'evergreen_trees', 'deciduous_trees',
                         'bare_earth', 'water', 'residential_buildings',
                         'non_residential_buildings', 'roads', 'sidewalks',
                         'crosswalks', 'major_thoroughfares', 'highways',
                         'railways', 'paved_parking_lots', 'unpaved_parking_lots',
                         'cars', 'trains', 'stadium_seats'],
        'sensor': 'ITRES',
        'ignored_labels': [0],
        'palette': None,
        'img_pth': '{}/Datasets/Houston/houston.npy'.format(get_path()),
        'gt_pth': {'train': '{}/Datasets/Houston/initial_gt.npy'.format(get_path()),
                   'val': '{}/Datasets/Houston/val_gt.npy'.format(get_path()),
                   'test': '{}/Datasets/Houston/test_gt.npy'.format(get_path()),
                   'labeled_pool': '{}/Datasets/Houston/labeled_pool.npy'.format(get_path())
        }
    },

    'IndianPines': {
        'label_values': ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
            "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
            "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
            "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers"],
        'sensor': 'AVIRIS',
        'ignored_labels': [0],
        'palette': None,
        'img_pth': '{}/Datasets/IndianPines/indian_pines.npy'.format(get_path()),
        'gt_pth': {'train': dict((i, '{}/Datasets/IndianPines/gt{}/initial_gt.npy'.format(get_path(), i)) for i in range(1, 6)),
                   'test': dict((i, '{}/Datasets/IndianPines/gt{}/test_gt.npy'.format(get_path(), i)) for i in range(1,6)),
                   'labeled_pool': dict((i, '{}/Datasets/IndianPines/gt{}/labeled_pool.npy'.format(get_path(), i)) for i in range(1,6))
        }
    },



  'PaviaU' : {'label_values':  ['Undefined', 'Asphalt', 'Meadows',
                                'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil',
                                'Bitumen', 'Self-Blocking bricks', 'Shadows'],
              'sensor': 'ROSIS',
              'ignored_labels': [0],
              'palette': None,
              'img_pth': '{}/Datasets/PaviaU/img.npy'.format(get_path()),
              'gt_pth': {'train': dict((i, '{}/Datasets/PaviaU/gt{}/initial_gt.npy'.format(get_path(), i)) for i in range(1,6)),
                         'test': dict((i, '{}/Datasets/PaviaU/gt{}/test_gt.npy'.format(get_path(), i)) for i in range(1,6)),
                         'labeled_pool': dict((i, '{}/Datasets/PaviaU/gt{}/labeled_pool.npy'.format(get_path(), i)) for i in range(1,6))
             }},


    'SmallPaviaU' : {'label_values':  ['Undefined', 'Asphalt', 'Meadows',
                                  'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil',
                                  'Bitumen', 'Self-Blocking bricks', 'Shadows'],
                'sensor': 'ROSIS',
                'ignored_labels': [0],
                'palette': None,
                'img_pth': '{}/Datasets/PaviaU/small_img.npy'.format(get_path()),
                'gt_pth': {'train': dict((i, '{}/Datasets/PaviaU/small_gt{}/initial_gt.npy'.format(get_path(), i)) for i in range(1,2)),
                           'test': dict((i, '{}/Datasets/PaviaU/small_gt{}/test_gt.npy'.format(get_path(), i)) for i in range(1,2)),
                           'labeled_pool': dict((i, '{}/Datasets/PaviaU/small_gt{}/labeled_pool.npy'.format(get_path(), i)) for i in range(1,2))
               }},

    'PaviaU_partial' : {'label_values':  ['Undefined', 'Asphalt', 'Meadows',
                                  'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil',
                                  'Bitumen', 'Self-Blocking bricks', 'Shadows'],
                'sensor': 'ROSIS',
                'ignored_labels': [0],
                'palette': None,
                'img_pth': '{}/Datasets/PaviaU/img.npy'.format(get_path()),
                'gt_pth': {'train': dict((i, '{}/Datasets/PaviaU_partial/gt{}/initial_gt.npy'.format(get_path(), i)) for i in range(1,6)),
                           'test': dict((i, '{}/Datasets/PaviaU_partial/gt{}/test_gt.npy'.format(get_path(), i)) for i in range(1,6)),
                           'labeled_pool': dict((i, '{}/Datasets/PaviaU_partial/gt{}/labeled_pool.npy'.format(get_path(), i)) for i in range(1,6))
               }},

   'PaviaUcoreset' : {'label_values':  ['Undefined', 'Asphalt', 'Meadows',
                                 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil',
                                 'Bitumen', 'Self-Blocking bricks', 'Shadows'],
               'sensor': 'ROSIS',
               'ignored_labels': [0],
               'palette': None,
               'img_pth': '{}/Datasets/PaviaU/img.npy'.format(get_path()),
               'gt_pth': {'train': dict((i, '{}/Datasets/PaviaU/gt{}/initial_gt.npy'.format(get_path(), i)) for i in range(1,6)),
                          'test': dict((i, '{}/Datasets/PaviaU/gt{}/test_gt.npy'.format(get_path(), i)) for i in range(1,6)),
                          'labeled_pool': dict((i, '{}/Datasets/PaviaU/gt{}/labeled_pool.npy'.format(get_path(), i)) for i in range(1,6))
              }},

  'PaviaC' : {'label_values':  ['Undefined', 'Water', 'Trees', 'Meadows',
                                'Self-Blocking Bricks', 'Bare Soil', 'Asphalt',
                                'Bitumen', 'Tiles', 'Shadows'],
              'sensor': 'ROSIS',
              'ignored_labels': [0],
              'palette': None,
              'img_pth': '/home/rothor/Documents/ONERA/Datasets/PaviaC/img.npy',
              'gt_pth': {'train':  '/home/rothor/Documents/ONERA/Datasets/PaviaC/train_gt.npy',
                         'val':    '/home/rothor/Documents/ONERA/Datasets/PaviaC/val_gt.npy',
                         'test':   '/home/rothor/Documents/ONERA/Datasets/PaviaC/test_gt.npy'}
               },

  # 'Mauzac' : {'label_values':  ['Untitled', 'VegetationShadow', 'HighVegetation', 'GroundVegetation','DryVegetation',
  #                               'BareSoil', 'WaterBody', 'Pool', 'PoolCover', 'Curbstone',' Tile', 'Asphalt', 'OtherShadow'],
  #             'sensor': 'FENIX',
  #             'ignored_labels': [0],
  #             'palette': [(0,0,0),(157,181,178),(21,113,69),(181,186,114),(249,203,64),(129,81,54),(63,142,252),
  #                         (0, 255, 255), (66,32,64), (194,173,180),(255,103,92), (0,6,153), (41,47,54)],
  #             'img_pth': '/home/rothor/Documents/ONERA/Datasets/Mauzac/train_base.npy',
  #             'gt_pth': {'train':  '/home/rothor/Documents/ONERA/Datasets/Mauzac/AL/gt.npy',
  #                        'val':    '/home/rothor/Documents/ONERA/Datasets/Mauzac/AL/gt.npy',
  #                        'test':   '/home/rothor/Documents/ONERA/Datasets/Mauzac/AL/gt.npy'}
  #              },

      'Mauzac' : {'label_values':  ['Untitled', 'HighVegetation', 'GroundVegetation','DryVegetation',
                                      'WaterBody', 'Tile', 'Asphalt'],
                  'sensor': 'FENIX',
                  'ignored_labels': [0],
                  'palette': [(0,0,0),(21,113,69),(181,186,114),(249,203,64),(63,142,252),
                              (255,103,92), (0,6,153)],
                  'img_pth': '{}/Datasets/Mauzac/train_base.npy'.format(get_path()),
                  'gt_pth': {'train':  {1: '{}/Datasets/Mauzac/initial_gt.npy'.format(get_path())},
                             'labeled_pool':    {1: '{}/Datasets/Mauzac/labeled_pool.npy'.format(get_path())},
                             'test':   {1: '{}/Datasets/Mauzac/test_gt.npy'.format(get_path())}}
                   },

    'toy_dataset' : {'label_values':  ['Untitled', 'Tile', 'DryVegetation'],
                'sensor': 'toy_sensor',
                'ignored_labels': [0],
                'palette': [(0,0,0), (227, 101, 91), (0, 83, 119)],
                'img_pth': '/home/rothor/Documents/ONERA/Datasets/full/Mauzac/toy_dataset/img.npy',
                'gt_pth': {'train':  {0: '{}/Datasets/full/Mauzac/toy_dataset/initial_gt.npy'.format(get_path())},
                           'labeled_pool':    {0: '{}/Datasets/full/Mauzac/toy_dataset/labeled_pool.npy'.format(get_path())},
                           'test':   {0: '{}/Datasets/full/Mauzac/toy_dataset/test_gt.npy'.format(get_path())}}
                 },

    'MauzacAsphaltTile' : {'label_values':  ['Untitled', 'Tile', 'Asphalt'],
                'sensor': 'FENIX',
                'ignored_labels': [0],
                'palette': [(0,0,0),(255,103,92),(0,6,153)],
                'img_pth': '/home/rothor/Documents/ONERA/Datasets/Mauzac/train_base.npy',
                'gt_pth': {'train':  '/home/rothor/Documents/ONERA/Datasets/Mauzac/geometry_exp/asphalt_tile_gt.npy',
                           'val':    '/home/rothor/Documents/ONERA/Datasets/Mauzac/geometry_exp/asphalt_tile_gt.npy',
                           'test':   '/home/rothor/Documents/ONERA/Datasets/Mauzac/geometry_exp/asphalt_tile_gt.npy'}
                 },

    'Toulouse' : {

    'label_values':  ['Untitled', 'VegetationShadow', 'HighVegetation', 'GroundVegetation', 'DryGroundVegetation',
                                  'BareSoil', 'WaterBody', 'Pool', 'PoolCover', 'Curbstone', ' Tile', 'Asphalt', 'OtherShadow'],
                'sensor': 'FENIX',
                'ignored_labels': [0],
                'palette': [(0,0,0),(157,181,178),(21,113,69),(181,186,114),(249,203,64),(129,81,54),(63,142,252),
                            (0, 255, 255), (66,32,64), (194,173,180),(255,103,92), (0,6,153), (41,47,54)],
                'img_pth': '/home/rothor/Documents/ONERA/Datasets/Toulouse/TLS_2019-02-13_11-11-57_atmcor.hdr',
                'gt_pth': {'train':  {1: '/home/rothor/Documents/ONERA/Datasets/Toulouse/residentiel/train_gt.npy'},
                           'val':    {1: '/home/rothor/Documents/ONERA/Datasets/Toulouse/residentiel/val_gt.npy'},
                           'test':   {1: '/home/rothor/Documents/ONERA/Datasets/Toulouse/residentiel/all_gt.npy'}}
                 },

  'KSC' : {'label_values':  ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Water"],
            'sensor': 'AVIRIS',
            'ignored_labels': [0],
            'palette': None,
            'img_pth': '/home/rothor/Documents/ONERA/Datasets/KSC/img.npy',
            'gt_pth': {'train':  '{}/Datasets/KSC/AL/train_gt.npy'.format(get_path()),
                        'val':    '{}/Datasets/KSC/AL/val_gt.npy'.format(get_path()),
                        'test':   '{}/Datasets/KSC/AL/test_gt.npy'.format(get_path()),
                        'labeled_pool': '{}/Datasets/KSC/AL/labeled_pool.npy'.format(get_path()) }
    }
  }


DATASETS_CONFIG['IndianPinesCoreset'] = DATASETS_CONFIG['IndianPines']
DATASETS_CONFIG['PaviaUBatchBald'] = DATASETS_CONFIG['PaviaU']
DATASETS_CONFIG['PaviaU_partial'] = DATASETS_CONFIG['PaviaU']
DATASETS_CONFIG['PaviaUV2'] = DATASETS_CONFIG['PaviaU']
DATASETS_CONFIG['IndianPinesPP'] = DATASETS_CONFIG['IndianPines']

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
