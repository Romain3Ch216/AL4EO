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

    def step(self):
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
        train_data = self.dataset.train_data

        if self.config['superpixels']:
            pool = self.dataset.pool_data()
            self.dataset.pool_segmentation_(pool[0], pool[1], self.queried_clusters)
            pool, cluster_ids, clusters = self.dataset.spectra, self.dataset.cluster_ids, self.dataset.clusters
            pool = pool, np.arange(pool.shape[0])
            ranks = self.query(self.model, pool, train_data)
            self.queried_clusters = ranks
            indices = []
            for rank in ranks :
                cluster_id = cluster_ids[rank]
                inds = np.where(cluster_id == clusters)
                random_ind = np.random.randint(low=0, high=len(inds[0]), size=int(self.config['n_random']))
                indices.extend(inds[0][random_ind])

            indices = np.array(indices).astype(int)
            self.coordinates = self.dataset.pool.coordinates.T[indices]

            if self.query.score is not None:
                corr = dict((i, self.query.score[i]) for i in range(len(self.query.score)))
                score = np.vectorize(corr.get)(clusters)
                score_map = np.zeros_like(self.dataset.train_gt.gt).astype(float)
                coord = tuple((self.dataset.pool.coordinates[0], self.dataset.pool.coordinates[1]))
                score_map[coord] = score

        else:
            pool = self.dataset.pool_data()
            ranks = self.query(self.model, pool, train_data)
            self.coordinates = self.dataset.pool.coordinates.T[ranks]
            score = self.query.score

        
        
        query_time = time.time() - start_query_time
        f.writelines(['{} {} {} {}\n'.format(self.step_, training_time, query_time, pool[0].shape[0])])
        f.close()

        self.history['coordinates'].extend(list(self.coordinates))
        self.step_ += 1

    def oracle(self):
        coordinates = tuple((self.coordinates[:,0], self.coordinates[:,1]))
        labels = self.dataset.pool.gt[coordinates]
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
        pkl.dump((self.history, self.classes, self.config),\
          open(os.path.join(self.res_dir, 'history_{}.pkl'.format(self.config['timestamp'])), 'wb'))

    def restore(self):
        raise NotImplementedError()
        # with open(self.config['restore'], 'rb') as f:
        #     self.dataset.train_gt, self.classes, self.history, self.history, self.config['timestamp'] = pkl.load(f)

        # self.step_ = self.history['iteration'][-1]
        # self.config['n_classes'] = self.n_classes
        # self.config['classes'] = np.arange(1, self.n_classes)
        # self.model, self.query, self.config = load_query(self.config, self.dataset)

    def init_step(self):
        self.config['pool_size'] = self.dataset.pool.size

