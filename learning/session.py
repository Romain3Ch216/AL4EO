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
        self.timestamp = time.strftime("%y_%m_%d_%H_%M")

        self.history = { 'iteration': [],
                         'train_loss': [],
                         'val_loss': [],
                         'train_accuracy': [],
                         'val_accuracy': [],
                         'train_IoU': [],
                         'val_IoU': [],
                         'batch_variance': [],
                         'sampling_time': [],
                         'training_time': []
                         }

        self.added = {
            'coordinates': [],
            'labels': []
        }

        self.init_classes()
        self.n_classes = len(self.classes.keys())
        self.n_px = config['n_px']
        self.res_dir = config['res_dir']

        if self.config['restore']:
            print('Restore history...')
            self.restore()

    def step(self):
        # pdb.set_trace()
        self.init_step()
        self.model.init_params()

        print('Training model...')
        start_time = time.time()
        self.model.train(self.dataset, self.config)
        training_time = time.time() - start_time

        print('Computing heuristic...')
        start_time = time.time()
        train_data = self.dataset.train_data
        if self.config['benchmark']:# or self.config['opening']:
            pool = self.dataset.pool_data
            #pdb.set_trace()
            ranks = self.query(self.model, pool, train_data)
            self.coordinates = self.dataset.pool.coordinates.T[ranks]
            score = self.query.score
        else:
            print('Clustering...')
            pool = self.dataset.pool_data()
            pool, cluster_ids, clusters = self.dataset.segmented_pool(pool[0], pool[1])
            pool = pool, np.arange(pool.shape[0])
            ranks = self.query(self.model, pool, train_data)
            indices = []
            for rank in ranks :
                cluster_id = cluster_ids[rank]
                inds = np.where(cluster_id == clusters)
                random_ind = np.random.randint(low=0, high=len(inds[0]), size=1)
                indices.extend(inds[0][random_ind])

            indices = np.array(indices).astype(int)
            self.coordinates = self.dataset.pool.coordinates.T[indices]

            corr = dict((i, self.query.score[i]) for i in range(len(self.query.score)))
            score = np.vectorize(corr.get)(clusters)

        score_map = np.zeros_like(self.dataset.train_gt.gt).astype(float)
        coord = tuple((self.dataset.pool.coordinates[0], self.dataset.pool.coordinates[1]))
        score_map[coord] = score
        import matplotlib.pyplot as plt
        plt.imshow(score_map)
        plt.show()
        sampling_time = time.time() - start_time

        self.added['coordinates'].extend(list(self.coordinates))
        self.step_ += 1
        # self.display(self.coordinates)
        self.update_history(self.model.history, self.coordinates, training_time, sampling_time)

        #if not self.config['benchmark']:
        self.save_query()

    def oracle(self):
        coordinates = tuple((self.coordinates[:,0], self.coordinates[:,1]))
        labels = self.dataset.pool.gt[coordinates]
        self.dataset.train_gt.add(coordinates, labels)
        added_labels = self.dataset.GT['labeled_pool'][coordinates]
        self.added['labels'].extend(added_labels)
        self.dataset.pool.remove(coordinates)
        self.query.hyperparams['classes'] = np.unique(self.dataset.train_gt())[1:]
        self.query.hyperparams['proportions'] = self.dataset.proportions
        self.config['classes'] = np.unique(self.dataset.train_gt())[1:]
        self.config['proportions'] = self.dataset.proportions

    def update_history(self, history, coordinates, training_time, sampling_time):
        history = dict((k, v[-1]) for k, v in history.items())

        self.history['iteration'].append(self.step_)
        self.history['train_loss'].append(round(history['train_loss'], 2))
        self.history['train_accuracy'].append(round(history['train_accuracy'], 2))
        self.history['train_IoU'].append(round(history['train_IoU'], 2))

        self.history['val_loss'].append(round(history['val_loss'],2))
        self.history['val_accuracy'].append(round(history['val_accuracy'],2))
        self.history['val_IoU'].append(round(history['val_IoU'],2))

        self.history['batch_variance'].append(round(self.batch_variance(coordinates), 2))
        self.history['sampling_time'].append(round(sampling_time/60, 2))
        self.history['training_time'].append(round(training_time/60, 2))

    def batch_variance(self, coordinates):
        coordinates = tuple((coordinates[:,0], coordinates[:,1]))
        spectra = self.dataset.IMG[coordinates]
        variance = np.sum(np.var(spectra, axis=0))
        return variance

    def display(self, coordinates):
        self.regions = np.zeros_like(self.dataset.GT['train'])
        self.coordinates = coordinates
        self.patch_id = 0
        self.patches, self.patch_coordinates = window(self.dataset.IMG, coordinates)
        # self.score, _ = window(self.query.score.reshape(self.dataset.IMG.shape[:-1]), coordinates)
        self.regions, _ = window(self.regions, coordinates)

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
        pkl.dump((self.dataset.train_gt, self.classes, self.added, self.history, self.timestamp, self.config),\
          open(os.path.join(self.res_dir, 'history_{}.pkl'.format(self.timestamp)), 'wb'))

    def restore(self):
        with open(self.config['restore'], 'rb') as f:
            self.dataset.train_gt, self.classes, self.added, self.history, self.timestamp = pkl.load(f)
        # self.dataset.label_values = [v['label'] for v in self.classes.values()]
        # self.n_classes = len(self.dataset.label_values)
        # self.config['n_classes'] = self.n_classes
        # self.config['classes'] = np.arange(1, self.n_classes)
        self.step_ = self.history['iteration'][-1]
        self.config['n_classes'] = self.n_classes
        self.config['classes'] = np.arange(1, self.n_classes)
        self.model, self.query, self.config = load_query(self.config, self.dataset)

    def init_step(self):
        # import pyro
        # if self.config['query'] in ['bald', 'batch_bald'] and self.step_ > 0:
            # pyro.nn.module.clear(self.model.model)
            # pyro.nn.module.clear(self.model.net)
        self.config['pool_size'] = self.dataset.pool.size

    def save_query(self):
        if 'scheduler' in self.config:
            del self.config['scheduler']
        if 'weights' in self.config:
            del self.config['weights']
        pkl.dump((self.history, self.classes, self.query.score, self.coordinates, self.dataset.train_gt, self.config, self.timestamp) ,\
            open(os.path.join(self.config['res_dir'], 'query_{}_step_{}.pkl'.format(self.timestamp, self.step_)), 'wb'))
