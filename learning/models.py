# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import torchnet as tnt

# Utils
import math
import numpy as np
import pdb
from tqdm import tqdm
import copy
from learning.utils import sliding_window, count_sliding_window, grouper
from sklearn.svm import SVC


# Metrics
from learning.metrics import mIou
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

#===============================================================================

class Classifier:
    """
    Generic class for classifiers
    """
    def __init__(self):
        self.best_metric = 0
        self.history = {
            'train_loss': [np.nan],
            'train_accuracy': [np.nan],
            'train_IoU': [np.nan],
            'train_kappa': [np.nan],
            'grad': [np.nan],
            'val_loss': [np.nan],
            'val_accuracy': [np.nan],
            'val_IoU': [np.nan],
            'val_kappa': [np.nan],
            'test_loss': [np.nan],
            'test_accuracy': [np.nan],
            'test_IoU': [np.nan],
            'test_kappa': [np.nan]
        }

    def train(self, dataset, config):
        # raise NotImplementedError
        return None

    def predict_probs(self, data_loader, config):
        raise NotImplementedError

    def init_params(self):
        return None


#===============================================================================
#                  Class for conventional neural networks
#===============================================================================

class NeuralNetwork(Classifier):
    def __init__(self, net, optimizer, criterion):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.softmax = nn.Softmax(dim=-1)

        self.best_metric = 0
        self.best_epoch = 0
        self.best_state = self.net.state_dict()

    def init_params(self):
        self.net.apply(self.net.weight_init)

    def train(self, dataset, hyperparams):
        train_data_loader, val_data_loader = dataset.load_data(dataset.img, dataset.train_gt())

        for epoch in range(1, hyperparams['epochs']+1):
            print('EPOCH {}/{}'.format(epoch, hyperparams['epochs']))

            self.net.train()
            self.net.to(hyperparams['device'])

            acc_meter      = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter     = tnt.meter.AverageValueMeter()
            grad_meter     = dict((depth, tnt.meter.AverageValueMeter()) for depth, _ in enumerate(self.net.parameters()))
            y_true, y_pred = [], []

            for batch_id, (spectra, y) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):

                y_true.extend(list(map(int, y)))
                spectra, y = spectra.to(hyperparams['device']), y.to(hyperparams['device'])

                self.optimizer.zero_grad()
                out = self.net(spectra)
                loss = self.criterion(out, y.long())

                loss.backward()
                self.optimizer.step()

                pred = out.detach()
                y_p = pred.argmax(dim=1).cpu().numpy()
                y_pred.extend(list(y_p))
                acc_meter.add(pred, y)
                loss_meter.add(loss.item())
                for depth, params in enumerate(self.net.parameters()):
                    if params.grad is not None:
                        grad_meter[depth].add(torch.mean(torch.abs(params.grad)).item())

            self.history['train_accuracy'].append(acc_meter.value()[0])
            self.history['train_loss'].append(loss_meter.value()[0])
            self.history['train_IoU'].append(mIou(y_true, y_pred, hyperparams['n_classes']))
            self.history['train_kappa'].append(cohen_kappa_score(y_true, y_pred))

            print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(\
            self.history['train_loss'][-1], self.history['train_accuracy'][-1], self.history['train_IoU'][-1]))

            # Validation

            y_true, y_pred = [], []
            acc_meter      = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter     = tnt.meter.AverageValueMeter()

            self.net.eval()

            for (spectra, y) in val_data_loader:
                y_true.extend(list(map(int, y)))
                spectra, y = spectra.to(hyperparams['device']), y.to(hyperparams['device'])

                with torch.no_grad():
                    prediction = self.net(spectra)
                    loss = self.criterion(prediction, y)

                acc_meter.add(prediction, y)
                loss_meter.add(loss.item())
                y_p = prediction.argmax(dim=1).cpu().numpy()
                y_pred.extend(list(y_p))

            if acc_meter.value()[0] > self.best_metric:
                self.best_metric = acc_meter.value()[0]
                self.best_epoch = epoch
                self.best_state = self.net.state_dict()

            self.history['val_accuracy'].append(acc_meter.value()[0])
            self.history['val_loss'].append(loss_meter.value()[0])
            self.history['val_IoU'].append(mIou(y_true, y_pred, hyperparams['n_classes']))
            self.history['val_kappa'].append(cohen_kappa_score(y_true, y_pred))

            print('Validation: Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(\
            self.history['val_loss'][-1], self.history['val_accuracy'][-1], self.history['val_IoU'][-1]))


    def predict_probs(self, data_loader, hyperparams):
        self.net.to(hyperparams['device'])
        self.net.load_state_dict(self.best_state)
        self.net.eval()
        probs = []
        for batch_id, (data, _) in enumerate(data_loader):
            data = data.to(hyperparams['device'])
            with torch.no_grad():
                probs.append(self.net(data).cpu())
        probs = torch.cat(probs)
        probs = self.softmax(probs)
        return probs

    def map(self, dataset, hyperparams):
        self.net.to(hyperparams['device'])
        self.net.load_state_dict(self.best_state)
        self.net.eval()
        img = torch.from_numpy(dataset.img)
        h, w, b = img.shape
        img = img.reshape(-1, b)
        img = img.to(hyperparams['device'])
        self.net.eval()
        with torch.no_grad():
            probs = self.net(img)
        preds = torch.argmax(probs, dim=-1)
        preds = preds.reshape(h, w)
        return preds.cpu().numpy()

    def evaluate(self, dataset, hyperparams):
        self.net.load_state_dict(self.best_state)

        test_data_loader = dataset.load_data(dataset.img, dataset.GT['test'], split=False)

        y_true, y_pred = [], []
        acc_meter      = tnt.meter.ClassErrorMeter(accuracy=True)

        self.net.eval()
        self.net.to(hyperparams['device'])

        for (spectra, y) in test_data_loader:
            y_true.extend(list(map(int, y)))
            spectra, y = spectra.to(hyperparams['device']), y.to(hyperparams['device'])

            with torch.no_grad():
                prediction = self.net(spectra)


            acc_meter.add(prediction, y)

            y_p = prediction.argmax(dim=1).cpu().numpy()
            y_pred.extend(list(y_p))

        metrics = {'OA': acc_meter.value()[0],
                   'mIoU': mIou(y_true, y_pred, hyperparams['n_classes']),
                   'Kappa': cohen_kappa_score(y_true, y_pred),
                   'cm': confusion_matrix(y_true, y_pred, labels=list(range(hyperparams['n_classes'])))
                   }


        return metrics


#===============================================================================
#                  Basic and state-of-the-art neural networks
#      Parts of code were taken from https://github.com/nshaud/DeepHyperX
#===============================================================================

class fnn(nn.Module):
    """
    Fully-connected network
    """

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout, seed=None):
        super(fnn, self).__init__()

        self.fc1 = nn.Linear(input_channels, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_classes)
        if seed:
            self.seed = seed
        else:
            self.seed = torch.randint(0, 100, (1,1)).item()
        self.apply(self.weight_init)

        self.dropout = nn.Dropout(p=dropout)

    def penalty(self, alpha):
        weights = self.fc1.weight
        L1_norm = torch.sum(torch.abs(weights))
        return alpha * L1_norm

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        #self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        #x = self.dropout(x)
        x = self.fc(x)
        return x

    def conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    def weight_init(self, m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, n_convs=20, kernel_size=None, pool_size=None, seed=None, dropout=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
           # [In our experiments, k1 is better to be [ceil](n1/9)]
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
           # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, n_convs, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        if seed:
            self.seed = seed
        else:
            self.seed = torch.randint(0, 100, (1,1)).item()

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#===============================================================================
#                               SVM Classifier
#===============================================================================

class SVM(Classifier):
    def __init__(self, svm_params={"kernel": "rbf", "gamma": 0.1, "C": 1000}):
        super().__init__()
        self.svm_params = svm_params
        self.svm_params['probability'] = True
        self.svm = None

    def train(self, dataset, hyperparams):
        X_train, y_train = dataset.train_data
        self.svm = SVC(**self.svm_params)
        self.svm.fit(X_train, y_train)

    def predict_probs(self, data_loader, hyperparams):
        probs = []
        for batch_id, (data, _) in enumerate(data_loader):
            probs_ = self.svm.predict_proba(data)
            probs.append(torch.from_numpy(probs_))
        probs = torch.cat(probs)
        probs = torch.softmax(probs, dim=-1)
        return probs

    def evaluate(self, dataset, hyperparams):
        X_test, y_true = dataset.test_data
        y_pred = self.svm.predict(X_test)

        metrics = {'OA': (y_pred == y_true).mean(),
                   'mIoU': mIou(y_true, y_pred, hyperparams['n_classes']),
                   'Kappa': cohen_kappa_score(y_true, y_pred),
                   'cm': confusion_matrix(y_true, y_pred, labels=list(range(hyperparams['n_classes'])))}


        return metrics

#===============================================================================
#                         Multi View Classifier
#===============================================================================

class MultiView(Classifier):
    def __init__(self, net, optimizer, criterion, n_views=4, min_width=30):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.softmax = nn.Softmax(dim=-1)
        self.n_views = n_views
        self.min_width = min_width
        self.history = {}
        self.view_history = {}
        for view_id in range(n_views):
            self.view_history[view_id] =  {
                'train_loss': [np.nan],
                'train_accuracy': [np.nan],
                'train_IoU': [np.nan],
                'train_kappa': [np.nan],
                'grad': [np.nan],
                'val_loss': [np.nan],
                'val_accuracy': [np.nan],
                'val_IoU': [np.nan],
                'val_kappa': [np.nan],
                'test_loss': [np.nan],
                'test_accuracy': [np.nan],
                'test_IoU': [np.nan],
                'test_kappa': [np.nan]
            }


    def init_params(self):
        return None

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def compute_views(self, img):
        """ Computes views according to the correlation matrix
        * Args:
            - img: npy array, HS img
            - n_views: number of views
            - min_width: minimum number of bands for a view
        * Output:
            - views: dict of npy array for each view
        """
        n_views = self.n_views
        min_width = self.min_width

        cov = np.cov(img.reshape(-1, img.shape[-1]).T)
        diag = np.diagonal(cov).reshape(-1,1)
        correlation_matrix = cov / diag**0.5 / (diag.T)**0.5
        correlations = np.diagonal(correlation_matrix, offset=1)

        ind = np.argsort(correlations)
        indices = np.zeros(n_views-1)
        i = 0
        while ind[i] < min_width:
            i += 1
        indices[0] = ind[i]
        i, j = 1, 1
        while indices[-1] == 0 and j < len(ind):
            ind_ = ind[j]
            if np.min(np.abs(indices - ind_)) > min_width:
                indices[i] = ind_
                i += 1
                j += 1
            else:
                j += 1

        assert j != len(ind), "Could not split the data in {} views".format(n_views)

        self.view_indices = np.sort(indices).astype(int)

    def create_views(self, data):
        views = {}
        n_views = self.n_views
        if len(data.shape) >= 3:
            views[0] = data[:,:,:self.view_indices[0]]
            for view_id in range(1, len(self.view_indices)):
                views[view_id] = data[:,:,self.view_indices[view_id-1] : self.view_indices[view_id]]
            views[n_views-1] = data[:,:,self.view_indices[-1]:]
        else:
            views[0] = data[:,:self.view_indices[0]]
            for view_id in range(1, len(self.view_indices)):
                views[view_id] = data[:,self.view_indices[view_id-1] : self.view_indices[view_id]]
            views[n_views-1] = data[:,self.view_indices[-1]:]
        return views

    def train(self, dataset, hyperparams):
        self.compute_views(dataset.img)
        views = self.create_views(dataset.img)
        self.nets = {}

        for view_id in views:
            self.nets[view_id] = copy.deepcopy(self.net)
            self.nets[view_id].__init__(views[view_id].shape[-1], dataset.n_classes)
            self.nets[view_id].apply(self.weight_init)
            self.nets[view_id].to(hyperparams['device'])

            train_data_loader, val_data_loader = dataset.load_data(views[view_id], dataset.train_gt.gt)

            for epoch in range(1, hyperparams['epochs']+1):
                print('EPOCH {}/{}'.format(epoch, hyperparams['epochs']))

                self.nets[view_id].train()

                acc_meter      = tnt.meter.ClassErrorMeter(accuracy=True)
                loss_meter     = tnt.meter.AverageValueMeter()
                grad_meter     = dict((depth, tnt.meter.AverageValueMeter()) for depth, _ in enumerate(self.nets[view_id].parameters()))
                y_true, y_pred = [], []

                for batch_id, (spectra, y) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):

                    y_true.extend(list(map(int, y)))
                    spectra, y = spectra.to(hyperparams['device']), y.to(hyperparams['device'])

                    self.optimizer.zero_grad()
                    out = self.nets[view_id](spectra)
                    loss = self.criterion(out, y.long())

                    loss.backward()
                    self.optimizer.step()

                    pred = out.detach()
                    y_p = pred.argmax(dim=1).cpu().numpy()
                    y_pred.extend(list(y_p))
                    acc_meter.add(pred, y)
                    loss_meter.add(loss.item())
                    for depth, params in enumerate(self.nets[view_id].parameters()):
                        if params.grad is not None:
                            grad_meter[depth].add(torch.mean(torch.abs(params.grad)).item())

                self.view_history[view_id]['train_accuracy'].append(acc_meter.value()[0])
                self.view_history[view_id]['train_loss'].append(loss_meter.value()[0])
                self.view_history[view_id]['train_IoU'].append(mIou(y_true, y_pred, hyperparams['n_classes']))
                self.view_history[view_id]['train_kappa'].append(cohen_kappa_score(y_true, y_pred))

                print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(\
                self.view_history[view_id]['train_loss'][-1], self.view_history[view_id]['train_accuracy'][-1], self.view_history[view_id]['train_IoU'][-1]))

                # Validation

                y_true, y_pred = [], []
                acc_meter      = tnt.meter.ClassErrorMeter(accuracy=True)
                loss_meter     = tnt.meter.AverageValueMeter()

                self.nets[view_id].eval()

                for (spectra, y) in val_data_loader:
                    y_true.extend(list(map(int, y)))
                    spectra, y = spectra.to(hyperparams['device']), y.to(hyperparams['device'])

                    with torch.no_grad():
                        prediction = self.nets[view_id](spectra)
                        loss = self.criterion(prediction, y)

                    acc_meter.add(prediction, y)
                    loss_meter.add(loss.item())
                    y_p = prediction.argmax(dim=1).cpu().numpy()
                    y_pred.extend(list(y_p))

                self.view_history[view_id]['val_accuracy'].append(acc_meter.value()[0])
                self.view_history[view_id]['val_loss'].append(loss_meter.value()[0])
                self.view_history[view_id]['val_IoU'].append(mIou(y_true, y_pred, hyperparams['n_classes']))
                self.view_history[view_id]['val_kappa'].append(cohen_kappa_score(y_true, y_pred))

                print('Validation: Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(\
                self.view_history[view_id]['val_loss'][-1], self.view_history[view_id]['val_accuracy'][-1], self.view_history[view_id]['val_IoU'][-1]))

        self.history['train_accuracy'] = np.mean(np.array([np.array(self.view_history[view_id]['train_accuracy']) for view_id in range(self.n_views)]), axis=0)
        self.history['train_loss'] = np.mean(np.array([np.array(self.view_history[view_id]['train_loss']) for view_id in range(self.n_views)]), axis=0)
        self.history['train_IoU'] = np.mean(np.array([np.array(self.view_history[view_id]['train_IoU']) for view_id in range(self.n_views)]), axis=0)
        self.history['train_kappa'] = np.mean(np.array([np.array(self.view_history[view_id]['train_kappa']) for view_id in range(self.n_views)]), axis=0)

        self.history['val_accuracy'] = np.mean(np.array([np.array(self.view_history[view_id]['val_accuracy']) for view_id in range(self.n_views)]), axis=0)
        self.history['val_loss'] = np.mean(np.array([np.array(self.view_history[view_id]['val_loss']) for view_id in range(self.n_views)]), axis=0)
        self.history['val_IoU'] = np.mean(np.array([np.array(self.view_history[view_id]['val_IoU']) for view_id in range(self.n_views)]), axis=0)
        self.history['val_kappa'] = np.mean(np.array([np.array(self.view_history[view_id]['val_kappa']) for view_id in range(self.n_views)]), axis=0)


    def view_loader(self, data_loader):
        for data_, _ in data_loader:
            yield self.create_views(data_)

    def predict_probs(self, data_loader, hyperparams):
        probs = torch.zeros((self.n_views, hyperparams['pool_size'], hyperparams['n_classes']))

        for batch_id, views in enumerate(self.view_loader(data_loader)):
            for view_id in views:
                net = self.nets[view_id]
                batch_size = hyperparams['batch_size']
                net.to(hyperparams['device'])
                net.eval()
                data = views[view_id].to(hyperparams['device'])

                with torch.no_grad():
                    probs[view_id, batch_id*batch_size:min(hyperparams['pool_size'], (batch_id+1)*batch_size),:] = \
                        net(data)

        probs = self.softmax(probs)
        n_views, N, n_classes = probs.shape
        probs = probs.reshape(N, n_classes, n_views)
        return probs

#===============================================================================
#              Classes for Variarional Adversarial Active Learning
#                 Credits to https://github.com/sinhasam/vaal
#                        Code was partially modified
#===============================================================================

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.encoder(x)
        return x.numel()

    def __init__(self, input_channels, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.input_channels = input_channels
        if input_channels <= 2:
            self.encoder = nn.Sequential(
                nn.Linear(2, 10),
                nn.ReLU(True),
                nn.Linear(10, 2)
            )
            self.feature_size = 2
            self.fc_mu = nn.Linear(self.feature_size, z_dim)
            self.fc_logvar = nn.Linear(self.feature_size, z_dim)
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, self.feature_size),
                nn.Linear(2, 10),
                nn.ReLU(True),
                nn.Linear(10, 2)
            )

        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 4, 11, bias=False),
                nn.BatchNorm1d(4),
                nn.ReLU(True),
                nn.Conv1d(4, 8, 11, bias=False),
                nn.BatchNorm1d(8),
                nn.ReLU(True),
                nn.Conv1d(8, 16, 11, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(True),
                nn.Conv1d(16, 32, 11, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
            )
            self.feature_size = self._get_final_flattened_size()
            self.fc_mu = nn.Linear(self.feature_size, z_dim)
            self.fc_logvar = nn.Linear(self.feature_size, z_dim)
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, self.feature_size),
                View((-1, 32, self.feature_size//32)),
                nn.ConvTranspose1d(32, 16, 11, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(True),
                nn.ConvTranspose1d(16, 8, 11, bias=False),
                nn.BatchNorm1d(8),
                nn.ReLU(True),
                nn.ConvTranspose1d(8, 4, 11, bias=False),
                nn.BatchNorm1d(4),
                nn.ReLU(True),
                nn.ConvTranspose1d(4, 1, 11),
            )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self._encode(x)
        z = z.view(-1, self.feature_size)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        x_recon = x_recon.squeeze(1)
        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z).squeeze(-1)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class VaalClassifier(Classifier):
    def __init__(self, vae, discriminator):
        super().__init__()
        self.vae = vae
        self.discriminator = discriminator

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def init_params(self):
        self.vae.weight_init()
        self.discriminator.weight_init()

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for sp, label in dataloader:
                    yield sp, label
        else:
            while True:
                for sp, _ in dataloader:
                    yield sp


    def train(self, dataset, hyperparams):
        self.train_iterations = (dataset.n_train * hyperparams['epochs']) // hyperparams['batch_size']
        lr_change = self.train_iterations // 4

        train_data_loader, val_data_loader = dataset.load_data(dataset.img, dataset.train_gt())
        labeled_data = self.read_data(train_data_loader)
        unlabeled_data = self.read_data(dataset.load_data(dataset.img, dataset.pool(), split=False), labels=False)

        optim_vae = optim.Adam(self.vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=5e-4)

        self.vae.train()
        self.discriminator.train()

        if hyperparams['device'] == 'cuda':
            print("Running on cuda...")
            self.vae.cuda()
            self.discriminator.cuda()

        best_acc = 0
        for iter_count in tqdm(range(self.train_iterations)):

            labeled_sp, labels = next(labeled_data)
            unlabeled_sp = next(unlabeled_data)

            if hyperparams['device'] == 'cuda':
                labeled_sp = labeled_sp.cuda()
                unlabeled_sp = unlabeled_sp.cuda()
                labels = labels.cuda()

            # VAE step
            for count in range(hyperparams['num_vae_steps']):
                recon, z, mu, logvar = self.vae(labeled_sp)
                unsup_loss = self.vae_loss(labeled_sp, recon, mu, logvar, hyperparams['beta'])
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = self.vae(unlabeled_sp)
                transductive_loss = self.vae_loss(unlabeled_sp,
                        unlab_recon, unlab_mu, unlab_logvar, hyperparams['beta'])

                labeled_preds = self.discriminator(mu)
                unlabeled_preds = self.discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_sp.size(0))
                unlab_real_preds = torch.ones(unlabeled_sp.size(0))

                if hyperparams['device'] == 'cuda':
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + hyperparams['adversary_param'] * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (hyperparams['num_vae_steps'] - 1):
                    labeled_sp, _ = next(labeled_data)
                    unlabeled_sp = next(unlabeled_data)

                    if hyperparams['device'] == 'cuda':
                        labeled_sp = labeled_sp.cuda()
                        unlabeled_sp = unlabeled_sp.cuda()
                        labels = labels.cuda()

            # Discriminator step
            for count in range(hyperparams['num_adv_steps']):
                with torch.no_grad():
                    _, _, mu, _ = self.vae(labeled_sp)
                    _, _, unlab_mu, _ = self.vae(unlabeled_sp)

                labeled_preds = self.discriminator(mu)
                unlabeled_preds = self.discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_sp.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_sp.size(0))

                if hyperparams['device'] == 'cuda':
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (hyperparams['num_adv_steps'] - 1):
                    labeled_sp, _ = next(labeled_data)
                    unlabeled_sp = next(unlabeled_data)

                    if hyperparams['device'] == 'cuda':
                        labeled_sp = labeled_sp.cuda()
                        unlabeled_sp = unlabeled_sp.cuda()
                        labels = labels.cuda()


            if iter_count % (self.train_iterations // 10) == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))


    def validate(self, task_model, loader, hyperparams):
        task_model.eval()
        total, correct = 0, 0
        y_true, y_pred = [], []
        for imgs, labels in loader:
            y_true.extend(labels)
            if hyperparams['device'] == 'cuda':
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        IoU = mIou(y_true, y_pred, hyperparams['n_classes'])
        return acc, IoU

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD


#===============================================================================
#                   Class for Bayesian Active Learning
#         Code from https://github.com/ElementAI/baal was partially modified
#===============================================================================

import sys
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from learning.utils import stack_in_memory
from learning.utils import map_on_tensor

def _stack_preds(out):
    if isinstance(out[0], Sequence):
        out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
    else:
        out = torch.stack(out, dim=-1)
    return out


class BayesianModelWrapper:
    """
    Wrapper created to ease the training/testing/loading.

    Args:
        model (nn.Module): The model to optimize.
        criterion (Callable): A loss function.
        replicate_in_memory (bool): Replicate in memory optional.
    """

    def __init__(self, model, criterion,
                 replicate_in_memory=False):
        self.model = model
        self.criterion = criterion
        self.metrics = dict()
        self.replicate_in_memory = replicate_in_memory
        self.softmax = nn.Softmax(dim=-2)

        self.history = {
            'train_loss': [np.nan],
            'train_accuracy': [np.nan],
            'train_IoU': [np.nan],
            'train_kappa': [np.nan],
            'grad': [np.nan],
            'val_loss': [np.nan],
            'val_accuracy': [np.nan],
            'val_IoU': [np.nan],
            'val_kappa': [np.nan],
            'test_loss': [np.nan],
            'test_accuracy': [np.nan],
            'test_IoU': [np.nan],
            'test_kappa': [np.nan]
        }

    def train(self, dataset, hyperparams):
        """
        Train for `epoch` epochs on a Dataset `dataset.

        Args:
            dataset (Dataset): Pytorch Dataset to be trained on.
            optimizer (optim.Optimizer): Optimizer to use.
            batch_size (int): The batch size used in the DataLoader.
            epoch (int): Number of epoch to train for.
            use_cuda (bool): Use cuda or not.
            workers (int): Number of workers for the multiprocessing.
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.

        Returns:
            The training history.
        """
        optimizer = hyperparams['optimizer']
        batch_size = hyperparams['batch_size']
        epoch = hyperparams['epochs']
        use_cuda = True if hyperparams['device'] == 'cuda' else False
        num_workers = 4
        collate_fn = None
        self.model.train()
        history = []
        data_loader = dataset.load_data(dataset.img, dataset.train_gt(), split=False)
        collate_fn = collate_fn or default_collate
        for _ in range(epoch):
            for data, target in data_loader:
                _ = self.train_on_batch(data, target, optimizer, hyperparams)

        optimizer.zero_grad()  # Assert that the gradient is flushed.
        return history

    def predict_on_data_loader(self, data_loader, hyperparams):
        """
        Use the model to predict on a dataset `iterations` time.

        Args:
            dataset (Dataset): Dataset to predict on.
            batch_size (int):  Batch size to use during prediction.
            iterations (int): Number of iterations per sample.
            use_cuda (bool): Use CUDA or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            half (bool): If True use half precision.

        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Generators [batch_size, n_classes, ..., n_iterations].
        """
        self.model.eval()
        use_cuda = True if hyperparams['device'] == 'cuda' else None
        iterations = hyperparams['num_samples']

        for idx, (data, _) in enumerate(tqdm(data_loader, total=len(data_loader), file=sys.stdout)):

            pred = self.predict_on_batch(data, hyperparams)
            pred = map_on_tensor(lambda x: x.detach(), pred)
            yield map_on_tensor(lambda x: x.cpu().numpy(), pred)

    def predict_probs(self, data_loader, hyperparams):
        preds = list(self.predict_on_data_loader(data_loader, hyperparams))

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            probs = torch.from_numpy(np.vstack(preds))
            probs = self.softmax(probs)
            return probs
        return [np.vstack(pr) for pr in zip(*preds)]

    def train_on_batch(self, data, target, optimizer, hyperparams):
        """
        Train the current model on a batch using `optimizer`.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            optimizer (optim.Optimizer): An optimizer.
            cuda (bool): Use CUDA or not.
            regularizer (Optional[Callable]): The loss regularization for training.


        Returns:
            Tensor, the loss computed from the criterion.
        """

        data, target = data.to(hyperparams['device']), target.to(hyperparams['device'])
        self.model.to(hyperparams['device'])
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss

    def predict_on_batch(self, data, hyperparams):
        """
        Get the model's prediction on a batch.

        Args:
            data (Tensor): The model input.
            iterations (int): Number of prediction to perform.
            cuda (bool): Use CUDA or not.

        Returns:
            Tensor, the loss computed from the criterion.
                    shape = {batch_size, nclass, n_iteration}.

        Raises:
            Raises RuntimeError if CUDA rans out of memory during data replication.
        """
        iterations = hyperparams['num_samples']
        with torch.no_grad():
            data = data.to(hyperparams['device'])
            self.model.to(hyperparams['device'])
            if self.replicate_in_memory:
                data = map_on_tensor(lambda d: stack_in_memory(d, iterations), data)
                try:
                    out = self.model(data)
                except RuntimeError as e:
                    raise RuntimeError(
                        '''CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
                    Use `replicate_in_memory=False` in order to reduce the memory requirements.
                    Note that there will be some speed trade-offs''') from e
                out = map_on_tensor(lambda o: o.view([iterations, -1, *o.size()[1:]]), out)
                out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)
            else:
                out = [self.model(data) for _ in range(iterations)]
                out = _stack_preds(out)
            return out

    def init_params(self):
        """Reset all *resetable* layers."""

        def reset(m):
            for m in self.model.modules():
                getattr(m, 'reset_parameters', lambda: None)()

        self.model.apply(reset)


#===============================================================================
#                       Random Forest Classifier
#     Parts of code were taken from https://github.com/ksenia-konyushkova/LAL
#===============================================================================
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class LalRegressor(Classifier):
    def __init__(self, nEstimators, oob_score=True, n_jobs=8, criterion="entropy"):
        super().__init__()
        self.RF = RandomForestClassifier(nEstimators, criterion=criterion, oob_score=oob_score, n_jobs=n_jobs)
        self.regressor = RandomForestRegressor(n_estimators=500, max_depth=10, max_features=5, oob_score=True, n_jobs=8)
        self.nEstimators = nEstimators
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.n_features = 6

    def split(self, img, train_gt, tau):
        L_tau = np.zeros_like(train_gt)
        U_tau = np.copy(train_gt)
        for class_id in np.unique(U_tau):
            coord = np.where(U_tau == class_id)
            indices = np.arange(coord[0].size)
            size = min(coord[0].size -1, int(tau*coord[0].size))
            size = max(size, 1)
            choices = np.random.choice(indices, size=size, replace=False)
            rows = coord[0][choices]
            cols = coord[1][choices]
            indices = tuple((rows, cols))
            L_tau[indices] = train_gt[indices]
            U_tau[indices] = 0

        L_mask = L_tau != 0
        U_mask = U_tau != 0
        L_tau = {'labels': L_tau[L_mask], 'data': img[L_mask]}
        U_tau = {'labels': U_tau[U_mask], 'data': img[U_mask]}
        return L_tau, U_tau

    def get_features(self, RF, test_data, n_labeled, config, train_labels, max_label):
        n_test = test_data.shape[0]
        probs = np.zeros((n_test, config['n_classes']-1, self.nEstimators))
        for i, tree in enumerate(RF.estimators_):
            probs[:,:,i] = self.map_labels(train_labels, tree.predict_proba(test_data), max_label)

        n_dim = test_data.shape[-1]
        f_1 = np.mean(probs, axis=-1)
        f_2 = np.std(probs, axis=-1)
        # - proportion of classes
        f_3 = config['proportions'].reshape(1, -1)
        f_3 = np.repeat(f_3, f_1.shape[0], axis=0)
        # the score estimated on out of bag estimate
        f_4 = RF.oob_score_*np.ones((f_1.shape[0]))
        # - coeficient of variance of feature importance
        f_5 = np.std(RF.feature_importances_/n_dim)*np.ones((f_1.shape[0]))
        # - estimate variance of forest by looking at avergae of variance of some predictions
        f_6 = np.mean(f_2, axis=0).reshape(1, -1)
        f_6 = np.repeat(f_6, f_1.shape[0], axis=0)
        # - compute the average depth of the trees in the forest
        f_7 = np.mean(np.array([tree.tree_.max_depth for tree in RF.estimators_]))*np.ones((f_1.shape[0]))
        # - number of already labelled datapoints
        f_8 = n_labeled*np.ones((f_1.shape[0]))
        LALfeatures = []
        for i in range(f_1.shape[0]):
            LALfeatures.append(list(f_1[i,:]) + list(f_2[i,:]) + list(f_3[i,:]) + list([f_4[i]]) + list([f_5[i]]) + list(f_6[i,:]) + list([f_7[i]]) + list([f_8[i]]))
        LALfeatures = np.array(LALfeatures)
        return LALfeatures

    def map_labels(self, labels, probs, n_classes):
        """
        Fills the gap when classes are missing in the training dataset
        """
        res = np.zeros((probs.shape[0], n_classes))
        res[:, labels-1] = probs
        if isinstance(res, type(torch.ones(1))):
            res = res.cpu()
        return res


    def DataMonteCarlo(self, L_tau, U_tau, M, config):
        # Sample M points from U_tau
        n_u_tau = len(U_tau['labels'])
        # indices = np.arange(n_u_tau)
        # choices = np.random.choice(indices, size=M, replace=False)
        choices_ = []
        classes = np.unique(U_tau['labels'])
        n_classes = len(classes)
        n = int(M/n_classes)
        for class_id in classes:
            coord = np.where(U_tau['labels'] == class_id)
            indices = np.arange(coord[0].size)
            size = min(coord[0].size, n)
            size = max(size, 1)
            choices = np.random.choice(indices, size=size, replace=False)
            indices = coord[0][choices]
            choices_.extend(list(indices))
        choices = np.array(choices_)
        U_tau['data'] = U_tau['data'][choices, :]
        U_tau['labels'] = U_tau['labels'][choices]

        RF = RandomForestClassifier(self.nEstimators, criterion=self.criterion, oob_score=self.oob_score, n_jobs=self.n_jobs)
        # Train a random forest classifier
        RF.fit(L_tau['data'], L_tau['labels'])
        # Estimate test loss
        n_test = len(U_tau['labels'])
        probs = RF.predict_proba(U_tau['data'])
        # pdb.set_trace()
        L_tau_labels = np.unique(L_tau['labels'])
        max_label = config['n_classes']-1
        probs = self.map_labels(L_tau_labels, probs, max_label)
        hot_labels = np.zeros((n_test, max_label))
        hot_labels[np.arange(n_test), U_tau['labels']-1] = 1
        MSE = np.mean((probs - hot_labels)**2, axis=1)
        # Compute the classification state parameters
        LALfeatures = self.get_features(RF, U_tau['data'], len(L_tau['labels']), config, L_tau_labels, max_label)

        gains_quality = []
        for i_x, (x, y) in tqdm(enumerate(zip(U_tau['data'], U_tau['labels']))):
            # Form a new labeled dataset
            x = x.reshape(1, -1)
            train_data = np.concatenate((L_tau['data'], x), axis=0)
            train_labels = np.concatenate((L_tau['labels'].reshape(1, -1), y.reshape(1, -1)), axis=1)
            train_labels = train_labels.reshape(-1)
            test_data = np.copy(U_tau['data'])
            test_data = np.delete(test_data, i_x, axis=0)
            test_labels = np.copy(U_tau['labels'])
            test_labels = np.delete(test_labels, i_x, axis=0)
            # Train a classifier
            RF = RandomForestClassifier(self.nEstimators, criterion=self.criterion, oob_score=self.oob_score, n_jobs=self.n_jobs)
            RF.fit(train_data, train_labels)
            # Estimate new test loss
            probs = RF.predict_proba(U_tau['data'])
            probs = self.map_labels(L_tau_labels, probs, max_label)
            mse = np.mean((probs - hot_labels)**2, axis=1)
            gains_quality.append(np.mean(MSE - mse))
        return LALfeatures, gains_quality

    def create_dataset(self, dataset, config):
        Q, tau, M = config['Q'], config['tau'], config['M']
        observations = None
        labels = []
        M_count = 0
        for tau_ in tau:
            for q in range(Q):
                L_tau, U_tau = self.split(dataset.img, dataset.train_gt(), tau=tau_)
                features, gains = self.DataMonteCarlo(L_tau, U_tau, M, config)
                if observations is None:
                    observations = features
                else:
                    observations = np.concatenate((observations, features), axis=0)
                labels.extend(gains)
        labels = np.array(labels)
        return observations, labels

    def train(self, dataset, config):
        print("Build dataset...")
        observations, labels = self.create_dataset(dataset, config)
        print("Train regressor...")
        self.regressor.fit(observations, labels)
        X, y = dataset.train_data
        self.RF.fit(X, y)
        self.n_labeled = dataset.train_gt.size

    def predict_probs(self, data_loader, hyperparams):
        pred_error_reduction = []

        for batch_id, (data, labels_) in enumerate(data_loader):
            labels = hyperparams['classes']
            max_label = hyperparams['n_classes']-1
            features = self.get_features(self.RF, data, self.n_labeled, hyperparams, labels, max_label)
            pred_error_reduction.append(torch.from_numpy(self.regressor.predict(features)))
        pred_error_reduction = torch.cat(pred_error_reduction).numpy()
        return pred_error_reduction
