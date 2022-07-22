from sklearn.utils import shuffle
import torch
import torch.utils.data as data
import numpy as np
import spectral
import spectral.io.envi as envi
import rasterio as rio
import rasterio.rio.insp
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

from skimage.morphology import closing, disk
from skimage.filters import sobel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.segmentation import slic

import seaborn as sns

from data.sensors import get_sensor, Sensor

import pdb

class Subset:
    """Generic class for a subset of the dataset"""
    def __init__(self, gt):
        """
        Args:
            gt: 2D array of labels
        """
        self.gt = gt

    @property
    def size(self):
        return np.sum(self.gt!=0)

    @property
    def coordinates(self):
        coordinates_ = np.where(self.gt)
        return np.array(coordinates_)

    @property
    def labels(self):
        mask = self.gt != 0
        return self.gt[mask]

    def __call__(self):
        return self.gt

class Pool(Subset):
    """Class for an unlabeled pool"""
    def __init__(self, gt):
        super().__init__(gt)

    def remove(self, coordinates):
        self.gt[coordinates] = 0


class TrainGt(Subset):
    """Class for the labeled pool / training dataset"""
    def __init__(self, gt):
        super().__init__(gt)

    def add(self, coordinates, labels):
        self.gt[coordinates] = labels


class Dataset:
    """ Class for an hyperspectral / multispectral dataset """

    def __init__(self, hyperparams, img_pth, gt_pth, palette, label_values, ignored_labels, sensor=None, n_bands=None, img_shape=None, rgb_bands=None):
        """
        Args:
            hyperparams: dict, hyperparameters
            img_pth: str, path to .npy or .hdr hyperspectral image
            gt_path: dict, paths to .npy ground truths
            palette: list, rgb colors
            label_values: list, class labels
            ignored_labels: list, class ids to ignore, by default [0]
            sensor: string, the sensor that acquired the data
        """

        #Metadata
        self.ignored_labels = ignored_labels
        self.label_values   = label_values
        self.n_classes      = len(label_values)
        if sensor:
            self.sensor_name    = sensor
            self.sensor         = get_sensor(sensor)
            self.rgb_bands      = self.sensor.rgb_bands
        else:
            self.rgb_bands      = rgb_bands
        self.hyperparams = hyperparams
        self.run = int(hyperparams['run'][-1])
        self.segmentation = None

        #Load image and ground truth
        type = img_pth[-3:]

        if type == 'npy':
            self.load_numpy(img_pth, gt_pth)
            self.n_bands = n_bands if n_bands else self.img.shape[-1] 
            self.img_shape = img_shape if img_shape else (self.img.shape[0], self.img.shape[1])
            self.rgb = self.hyper2rgb(self.IMG, self.rgb_bands)
            self.type = 'npy'
        if img_pth[-4:] == 'tiff':
            type = 'tiff'
            self.load_Hdr(img_pth, gt_pth)
            self.img_pth = img_pth
            self.type = 'hdr'

        self.create_palette(palette)

        for set_ in self.GT:
            self.GT[set_] = self.GT[set_].astype(int)

        self.train_gt = TrainGt(self.GT['train'])

        if 'labeled_pool' in self.GT:
            self.pool = Pool(self.GT['labeled_pool'])
        else:
            mask = (self.GT['train'] == 0) #+ self.GT['val'] + self.GT['test']) == 0
            #mask[self.nan_mask] = False
            self.pool = Pool(mask)

        #self.n_px_per_class = self.n_px_per_class()

        if len(hyperparams['remove']) > 0:
            self.remove()

    def load_Hdr(self, img_pth, gt_pth): #clément
        with rio.open(img_pth) as src:
            bbl = src.tags(ns=src.driver)['bbl'].replace(' ', '').replace('{', '').replace('}', '').split(',')
            bbl = np.array(list(map(int, bbl)), dtype=int)
            bbl = tuple(np.where(bbl != 0)[0] + 1)
            self.n_bands = len(bbl)
            self.img_shape = src.height, src.width
            img_transform = src.transform
            img_crs = src.crs
            self.img_min, self.img_max, _ = rasterio.rio.insp.stats(src.read(bbl))

        self.GT = {}
        for base, pth in gt_pth.items():
            with rio.open(pth[self.run]) as src:
                if src.transform == img_transform and src.crs == img_crs and src.height == self.img_shape[0] and src.width == self.img_shape[1]:
                    gt = src.read(1).astype(np.uint8)
                else:
                    gt = np.zeros(self.img_shape, dtype=np.uint8)
                    reproject(
                        source=src.read(1),
                        destination=gt,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=img_transform,
                        dst_crs=img_crs,
                        resampling=Resampling.nearest)
            self.GT[base] = gt

    def load_numpy(self, img_pth, gt_pth, normalization=True, copy=True):
        self.img = np.load(img_pth)

        if self.hyperparams['edge_detection'] > 0:
            print("Apply edge detection...")
            self.edge_detection_(threshold=self.hyperparams['edge_detection'])

        #Filters NaN values
        nan_mask = np.isnan(self.img.sum(axis=-1))
        self.nan_mask = nan_mask
        if np.count_nonzero(nan_mask) > 0:
           print("""Warning: NaN have been found in the data.
                    It is preferable to remove them beforehand.
                    Learning on NaN data is disabled.""")
        self.img[nan_mask] = 0
        self.GT = {}
        for base, gt in gt_pth.items():
            gt = np.load(gt[self.run])
            gt[nan_mask] = 0
            self.GT[base] = gt

        if normalization:
            self.img = np.asarray(self.img, dtype='float32')
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        self.img = self.img
        self.IMG = np.copy(self.img)

    def load_hdr(self, img_pth, gt_pth, normalization=True, copy=True):
        raise NotImplementedError()

    def remove(self):
        for class_id in self.hyperparams['remove']:
            mask = self.train_gt() == class_id
            self.train_gt.gt[mask] = 0
            self.pool.gt[mask] = class_id

    def edge_detection_(self, threshold=0.1, radius=1):
        edges = sobel(np.mean(self.img, axis=-1))
        mask = edges > threshold
        mask = closing(mask, disk(radius))
        self.img[mask] = np.full_like(self.img[0,0,:], np.nan)

    def toy_example(self):
        self.img = self.img[:20, :20 ,:]
        self.IMG = np.copy(self.img)
        self.pool.img = self.img
        self.train_gt.img = self.img

        self.train_gt.gt = np.random.randint(0, self.n_classes, size=(20,20))
        self.pool.gt = np.random.randint(0, self.n_classes, size=(20,20))
        mask = self.pool.gt != 0
        self.train_gt.gt[mask] = 0

    def load_data(self, data_, gt, split=True):
        data_ = HyperX(data_, gt, **self.hyperparams)
        use_cuda = self.hyperparams['device'] == 'cuda'
        N = len(data_)
        if split:
            train_dataset, val_dataset = data.random_split(data_, [int(0.95*N), N - int(0.95*N)])
            train_loader  = data.DataLoader(train_dataset, shuffle=True,
                                      batch_size=self.hyperparams['batch_size'], pin_memory=use_cuda)
            val_loader  = data.DataLoader(val_dataset, shuffle=True,
                                      batch_size=self.hyperparams['batch_size'], pin_memory=use_cuda)
            return train_loader, val_loader
        else:
            loader  = data.DataLoader(data_, shuffle=False,
                                      batch_size=self.hyperparams['batch_size'], pin_memory=use_cuda)
            return loader

    def load_Hdrdata(self, gt, split=True, shuffle=True):
        data_ = HyperHdrX(self.img_pth, gt, self.img_min, self.img_max, shuffle, **self.hyperparams)
        use_cuda = self.hyperparams['device'] == 'cuda'
        N = len(data_)

        if split:
            indices = np.arange(N)
            split_indice = int(0.95*N)
            train_indices, val_indices = indices[:split_indice], indices[split_indice:]
            train_sampler = SubsetSampler(train_indices)
            val_sampler = SubsetSampler(val_indices)
            train_loader  = data.DataLoader(data_, sampler=train_sampler,
                                      batch_size=self.hyperparams['batch_size'], pin_memory=use_cuda)
            val_loader  = data.DataLoader(data_, sampler = val_sampler,
                                      batch_size=self.hyperparams['batch_size'], pin_memory=use_cuda)
            return train_loader, val_loader
        else:
            loader  = data.DataLoader(data_, batch_size=self.hyperparams['batch_size'], pin_memory=use_cuda)
            return loader

    def data(self, gt):
        mask = gt != 0
        return self.img[mask]

    def segmentation_(self, n_segments, compactness, apply_pca=True):
        if apply_pca:
            pca = PCA(n_components=3)
            img = pca.fit_transform(self.img.reshape(-1, self.img.shape[-1]))
            img = img.reshape(self.img.shape[0], self.img.shape[1], -1)
        else:
            img = np.mean(self.img, axis=-1)
        mask = (self.pool.gt != 0)
        mask = np.array(mask, dtype=np.bool)
        self.segmentation = slic(img, n_segments=int(n_segments), compactness=int(compactness), mask=mask)

    def pool_segmentation_(self, x_pool, y_pool, queried_clusters=None, min_size=0):
        clusters = self.segmentation[self.pool.gt != 0]

        if queried_clusters is None:
            self.cluster_ids = np.unique(clusters)
            self.spectra = np.zeros((len(self.cluster_ids), self.n_bands))
            self.mask = np.ones_like(self.cluster_ids).astype(np.bool)
            for j, cluster_id in enumerate(self.cluster_ids):
                if (clusters == cluster_id).sum() < min_size:
                    self.mask[list(self.cluster_ids).index(cluster_id)] = False
                self.spectra[list(self.cluster_ids).index(cluster_id),:] = np.mean(x_pool[clusters==cluster_id], axis=0)
        else:
            for j, cluster_id in enumerate(queried_clusters):
                cluster_id = self.cluster_ids[cluster_id]
                if (clusters == cluster_id).sum() == 0 or (clusters == cluster_id).sum() < min_size:
                    self.mask[list(self.cluster_ids).index(cluster_id)] = False
                else:
                    self.spectra[list(self.cluster_ids).index(cluster_id),:] = np.mean(x_pool[clusters==cluster_id], axis=0)

        self.cluster_ids = self.cluster_ids[self.mask]
        self.spectra = np.array(self.spectra, dtype=np.float32)
        self.spectra = self.spectra[self.mask]
        self.mask = self.mask[self.mask]
        self.clusters = clusters

    def pool_dataHdr(self):
        mask = self.pool() != 0
        nb_nonzero = np.count_nonzero(mask)
        data = np.zeros((nb_nonzero, self.n_bands), dtype="float32")
        k=0
        with rio.open(self.img_pth) as src:
            assert len(set(src.block_shapes)) == 1
            for _, window in src.block_windows(1):
                block = src.read(window=window)
                for i in range(window.row_off, window.row_off+window.height):
                    for j in range(window.col_off, window.col_off+window.width):
                        if mask[j,i]:
                            data[k,:] = block[:,i-window.row_off,j-window.col_off]
                            k+=1
        assert k == nb_nonzero
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        labels = self.pool.labels
        return data, labels

    @property
    def train_data(self):
        data = self.data(self.train_gt())
        labels = self.train_gt.labels
        return data, labels

    @property
    def pool_data(self):
        data = self.data(self.pool())
        labels = self.pool.labels
        return data, labels

    @property
    def test_data(self):
        mask = self.GT['test'] != 0
        labels = self.GT['test'][mask]
        data = self.img[mask]
        return data, labels

    @property
    def labeled_(self):
        return (self.train_gt.gt + self.pool.gt).flatten()

    @property
    def n_train(self):
        return self.train_gt.size

    @property
    def proportions(self):
        gt = np.zeros_like(self.train_gt.gt)
        for set_ in self.GT:
            gt += self.GT[set_]
        prop = np.zeros(self.n_classes-1)
        n = np.sum(gt != 0)
        for class_id in np.unique(gt):
            if class_id != 0:
                prop[class_id-1] = round(np.sum(gt == class_id) / n *100, 1)
        return prop

        
    def n_px_per_class(self):
        gt = np.zeros_like(self.train_gt.gt)
        gt_ = np.ones_like(self.train_gt.gt)
        for set_ in self.GT:
            gt += self.GT[set_]
        gt = gt.astype(int)
        n_px = np.zeros(self.n_classes-1)
        n = np.sum(gt != 0)
        for class_id in np.unique(gt):
            if class_id != 0:
                n_px[class_id-1] = np.sum(gt == class_id)
        return n_px
        
    def create_palette(self, palette):
        # Generate color palette in rgb and hex format
        if palette is None:
            self.palette = {'rgb': {0: (0,0,0)}, 'HEX': {0: '#000'}}
            for k, color in enumerate(sns.color_palette("hls", len(self.label_values) - 1)):
                rgb = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
                self.palette['rgb'][k + 1] = rgb
                self.palette['HEX'][k + 1] = '#%02x%02x%02x' % rgb
        else:
            self.palette = {'HEX': {}}
            self.palette['rgb'] = dict(zip(list(range(len(self.label_values))), palette))
            for class_id, color in self.palette['rgb'].items():
                self.palette['HEX'][class_id] = '#%02x%02x%02x' % color

    @property
    def mean_spectra_(self):
        mean_spectra = {}
        for class_id in range(1, self.n_classes):
            mean_spectra[class_id] = np.mean(self.img[self.train_gt() == class_id], axis=0)
        return mean_spectra

    @staticmethod
    def hyper2rgb(img, bands):
        """Convert hyperspectral cube to a rgb image.
        Args:
            img: HS npy cube
            bands: tuple of rgb bands
        Returns:
            npy rgb array
        """
        rgb = spectral.get_rgb(img, bands)
        rgb /= np.max(rgb)
        rgb = np.asarray(255 * rgb, dtype='uint8')
        return rgb

    @staticmethod
    def split_spectra(img,mask):
        """
        Args:
            img: hyperspectral image
            mask: list of bands to mask

        Returns:
            sp: a dictionary containing the continuous part of the hyperspectral cube
        """
        k = 0
        q = 0
        sp = {}
        if mask is not None:
            for i in range(len(mask)-1):
                if mask[i+1] != mask[i] and mask[i+1]==0:
                    sp['{}:{}'.format(k,i+1-q+k)] = img[:,:,k:i+1-q+k]
                    k = i+1-q+k
                if mask[i+1] != mask[i] and mask[i+1]==1:
                    q = i+1
        else:
            sp[0] = img
        return sp

    def smooth_spectra(self,sigma):
        """Apply a gaussian filter to the spectra
        Args:
            sigma: gaussian standard deviation
        """
        from scipy.ndimage import gaussian_filter
        kernel_size = 0

        img = self.img
        sp = self.split_spectra(img,self.mask_bands)
        b = img.shape[-1]
        res = np.zeros((img.shape[0],img.shape[1],b))
        k   = 0
        for key, spectrums in sp.items():
            m,n,b = spectrums.shape
            spectrums = spectrums.reshape(m*n,b)
            spectrums = gaussian_filter(spectrums, sigma=sigma)
            res[:,:,k:b+k] = spectrums.reshape(m,n,b)
            k = b+k
        self.img = np.asarray(res, dtype='float32')

class SubsetSampler(data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)    

class HyperHdrX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral dataset
        Credits to https://github.com/nshaud/DeepHyperX"""

    def __init__(self, data_pth, gt, data_min, data_max, shuffle, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            hyperparams : dict that includes:
                dataset: str, name of the dataset
                patch_size: int, size of the spatial neighbourhood
                ignored_labels: list, class ids to ignore
        """
        super(HyperHdrX, self).__init__()
        #self.data_pth = data_pth
        self.label = gt
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.data_min = data_min
        self.data_max = data_max
        self.data_src = rio.open(data_pth)
        self.shuffle = shuffle
        
        bbl = self.data_src.tags(ns=self.data_src.driver)['bbl'].replace(' ', '').replace('{', '').replace('}', '').split(',')
        bbl = np.array(list(map(int, bbl)), dtype=int)
        self.bbl_index = tuple(np.where(bbl != 0)[0] + 1)

        block_hw = 2,4
        bh = gt.shape[0] // block_hw[0]
        bw = gt.shape[1] // block_hw[1]
        self.blocks_slices = []
        for i in range(block_hw[0]):
            for j in range(block_hw[1]):
                slice_x = (i*bh, gt.shape[0]) if i == block_hw[0]-1 else (i*bh, (i+1)*bh)
                slice_y = (j*bw, gt.shape[1]) if j == block_hw[1]-1 else (j*bw, (j+1)*bw)
                self.blocks_slices.append((slice_x, slice_y))

        mask = np.ones_like(gt)
        for l in self.ignored_labels:
            mask[gt == l] = 0

        x_pos, y_pos = np.nonzero(mask)
        print(len(x_pos))
        p = self.patch_size // 2
        self.indices = [[] for _ in range(block_hw[0]*block_hw[1])]
        for x, y in zip(x_pos, y_pos):
            block_indice_x = x // bh
            block_indice_y = y // bw
            block_indice_x = block_indice_x if block_indice_x < block_hw[0] else block_indice_x-1
            block_indice_y = block_indice_y if block_indice_y < block_hw[1] else block_indice_y-1
            block_indice = block_indice_x*block_hw[1] + block_indice_y
            block_slice = self.blocks_slices[block_indice]
            if p > 0:
                if (x > p + block_slice[0][0] and x < block_slice[0][1] - p and 
                y > p + block_slice[1][0] and y < block_slice[1][1] - p and self.label[x, y] != 0):
                    self.indices[block_indice].append((x, y))
            else:
                self.indices[block_indice].append((x, y))

        i = 0
        while(i < len(self.indices)):
            if self.indices[i] == []:
                del self.blocks_slices[i]
                del self.indices[i]
            else:
                i+=1

        self.blocks_slices = np.array(self.blocks_slices, dtype=tuple)
        self.indices = np.array(self.indices, dtype=list)

    def closeDataFile(self):
        self.data_src.close()

    def __len__(self):
        return sum([len(self.indices[i]) for i in range(len(self.indices))])

    def __getitem__(self, i):
        
        if i == 0:
            if self.shuffle:
                #shuffle pixels in each blocks and blocks
                for indice in self.indices:
                    np.random.shuffle(indice)
                random_idx = np.arange(len(self.indices))
                np.random.shuffle(random_idx)
                self.blocks_slices[random_idx]
                self.indices[random_idx]

            self.block_index = 0
            self.data_block = self.data_src.read(self.bbl_index, window=Window.from_slices(tuple(self.blocks_slices[self.block_index][0]), tuple(self.blocks_slices[self.block_index][1])))
            self.len_last = 0
            self.len_curr = len(self.indices[self.block_index])
        else:
            if i >= self.len_curr:
                self.block_index += 1
                self.data_block = self.data_src.read(self.bbl_index, window=Window.from_slices(tuple(self.blocks_slices[self.block_index][0]), tuple(self.blocks_slices[self.block_index][1])))
                self.len_last = self.len_curr
                self.len_curr += len(self.indices[self.block_index])

        x, y = self.indices[self.block_index][i-self.len_last]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        label = self.label[x1:x2, y1:y2]

        x1, x2 = x1 - self.blocks_slices[self.block_index][0][0], x2 - self.blocks_slices[self.block_index][0][0]
        y1, y2 = y1 - self.blocks_slices[self.block_index][1][0], y2 - self.blocks_slices[self.block_index][1][0]

        data = self.data_block[:, x1:x2, y1:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data), dtype="float32")
        #normalize data
        data = (data - self.data_min) / (self.data_max - self.data_min)
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label

class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral dataset
        Credits to https://github.com/nshaud/DeepHyperX"""

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            hyperparams : dict that includes:
                dataset: str, name of the dataset
                patch_size: int, size of the spatial neighbourhood
                ignored_labels: list, class ids to ignore
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams["dataset"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])

        mask = np.ones_like(gt)
        for l in self.ignored_labels:
            mask[gt == l] = 0

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        if p > 0:
            self.indices = np.array(
                [
                    (x, y)
                    for x, y in zip(x_pos, y_pos)
                    if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
                ]
            )
        else:
            self.indices = np.array(
                [
                    (x, y)
                    for x, y in zip(x_pos, y_pos)
                ]
            )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label