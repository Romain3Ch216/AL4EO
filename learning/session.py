# Query
from learning.query import load_query

# Utils
from learning.utils import *
import numpy as np
import os
import pdb
import pprint
import time
import pickle as pkl
import pdb
from learning.query import load_query

class ActiveLearningFramework:
    def __init__(self, dataset, model, query, config):
        self.dataset = dataset
        self.model = model
        self.query = query
        self.config = config

        self.step_ = 0
        if 'timestamp' not in self.config:
            self.config['timestamp'] = time.strftime("%y_%m_%d_%H_%M")

        self.history = { 'coordinates': [],
                         'labels': []
                        }

        self.init_classes()
        self.n_classes = len(self.classes.keys())
        self.n_px = config['n_px']
        self.res_dir = config['res_dir']
        self.queried_clusters = None
        self.score_map = None

        if 'restore' in self.config:
            print('Restore history...')
            self.restore()

    def step(self, bounding_box):
        self.init_step()
        self.model.init_params()

        f = open(self.config['res_dir'] + '/log.txt', 'a')
        f.writelines(['step training query pool_size\n'])

        print('Training model...')
        start_time = time.time()
        self.model.train(self.dataset, self.config)
        training_time = time.time() - start_time

        print('Computing heuristic...')
        start_query_time = time.time()
        #train_data = self.dataset.train_data
        train_data = None

        pool = self.dataset.load_data(self.dataset.pool, shuffle=False, split=False, bounding_box=bounding_box)

        self.coordinates = self.query(self.model, pool, train_data) # This is the coordinates of the selected pixels
        score = self.query.score
        coords = self.query.coords # This the coordinates of the pool

        # # To show the score
        # import matplotlib.pyplot as plt 
        # map_ = np.zeros(self.dataset.img_shape)
        # coord = tuple((coords[:,0], coords[:,1]))
        # map_[coord] = score
        # fig = plt.figure()
        # plt.imshow(map_)
        # plt.show()

        # import pdb 
        # pdb.set_trace()
        
        query_time = time.time() - start_query_time
        f.writelines(['{} {} {} {}\n'.format(self.step_, training_time, query_time, self.dataset.pool.size)])
        f.close()
 
        self.history['coordinates'].extend(list(self.coordinates))
        self.step_ += 1

    def oracle(self):
        coordinates = tuple((self.coordinates[:,0], self.coordinates[:,1]))
        labels = self.dataset.pool.data[coordinates]
        self.dataset.train_gt.add(coordinates, labels)
        added_labels = self.dataset.GT['labeled_pool'][coordinates]
        self.history['labels'].extend(added_labels)
        self.dataset.pool.remove(coordinates)
        updated_classes_ = np.unique(self.dataset.train_gt())[1:]
        self.query.hyperparams['classes'] = updated_classes_
        self.config['classes'] = updated_classes_

    def init_classes(self):
        self.classes = {}
        for class_id, class_label in enumerate(self.dataset.label_values):
            self.classes[class_id] = {}

        for class_id, class_label in enumerate(self.dataset.label_values):
            self.classes[class_id]['label'] = class_label
            self.classes[class_id]['nb_px'] = np.sum(self.dataset.train_gt == class_id)
            self.classes[class_id]['added_px'] = 0
            self.classes[class_id]['pseudo_labels'] = 0

    def update_classes(self, new_label_id):
        self.classes[new_label_id]['added_px'] += 1

    def save(self):
        if 'scheduler' in self.config:
            del self.config['scheduler']
        if 'weights' in self.config:
            del self.config['weights']
        if 'step' in self.config:
            path = os.path.join(self.res_dir, 'history_{}_step_{}.pkl'.format(self.config['timestamp'], self.config['step']))
            pkl.dump((self.history, self.classes, self.config), open(path, 'wb'))
        else:
            path = os.path.join(self.res_dir, 'history_{}.pkl'.format(self.config['timestamp']))
            pkl.dump((self.history, self.classes, self.config), open(path, 'wb'))
        return path

    def restore(self):
        with open(self.config['restore'], 'rb') as f:
            self.dataset.train_gt, self.classes, self.history, self.config = pkl.load(f)

        self.dataset.label_values = [item['label'] for item in self.classes.values()]
        n_classes = len(self.dataset.label_values)
        self.dataset.n_classes = n_classes
        self.config['n_classes'] = n_classes
        self.model, self.query, self.config = load_query(self.config, self.dataset)


    def init_step(self):
        self.config['pool_size'] = self.dataset.pool.size
