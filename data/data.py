import torch
import torch.utils.data as data
import numpy as np
import rasterio as rio
import rasterio.rio.insp
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from skimage.segmentation import slic
import pdb  

class Subset:
    """Generic class for a subset of the dataset"""
    def __init__(self, gt):
        """
        Args:
            gt: 2D array of labels
        """
        self.data = gt

    @property
    def size(self):
        return np.sum(self.data!=0)

    @property
    def coordinates(self):
        coordinates_ = np.where(self.data)
        return np.array(coordinates_)

    @property
    def labels(self):
        mask = self.data != 0
        return self.data[mask]

    def __call__(self):
        return self.data

class Pool(Subset):
    """Class for an unlabeled pool"""
    def __init__(self, gt):
        super().__init__(gt)

    def remove(self, coordinates):
        self.data[coordinates] = 0


class TrainGt(Subset):
    """Class for the labeled pool / training dataset"""
    def __init__(self, gt):
        super().__init__(gt)

    def add(self, coordinates, labels):
        self.data[coordinates] = labels


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
        self.img_pth = img_pth 
        self.segmentation_file = ('/').join(img_pth.split('/')[:-1]) + '/segmentation.tif'
        self.gt_pth = gt_pth
        self.ignored_labels = ignored_labels
        self.label_values   = label_values
        self.n_classes      = len(label_values)
        self.rgb_bands      = rgb_bands     
        self.hyperparams = hyperparams

        data_src = rio.open(self.img_pth)
        bbl = data_src.tags(ns=data_src.driver)['bbl'].replace(' ', '').replace('{', '').replace('}', '').split(',')
        bbl = np.array(list(map(int, bbl)), dtype=int)
        self.bbl_index = tuple(np.where(bbl != 0)[0] + 1)

        if img_pth[-4:] == 'tiff':
            type = 'tiff'
            self.load_geo_ref_img(img_pth, gt_pth)
            self.img_pth = img_pth
            self.type = 'hdr'
        elif img_pth[-3:] == 'bsq':
            type = 'bsq'
            self.load_geo_ref_img(img_pth, gt_pth)
            self.img_pth = img_pth
            self.type = 'hdr'
        else:
            raise NotImplementedError("Only .tiff file format is accepted.")


    #Load geo tiff gt image and image data metadata and reproject gt 
    def load_geo_ref_img(self, img_pth, gt_pth): 
        #load metadata of image data (bad bands, n_bands, shape, min, max...)
        with rio.open(img_pth) as src:
            bbl = src.tags(ns=src.driver)['bbl'].replace(' ', '').replace('{', '').replace('}', '').split(',')
            bbl = np.array(list(map(int, bbl)), dtype=int)
            bbl = tuple(np.where(bbl != 0)[0] + 1)
            self.n_bands = len(bbl)
            self.img_shape = src.height, src.width
            img_transform = src.transform
            img_crs = src.crs
            self.img_min, self.img_max, _ = rasterio.rio.insp.stats(src.read(bbl))

        #load gt image in memory and reproject it to the same shape as image data 

        with rio.open(gt_pth) as src:
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
            self.train_gt = TrainGt(gt)

        mask = self.train_gt.data == 0 
        self.pool = Pool(mask)
            

    #Create dataloader for geo referenced image, if shuffle == True, indice are shuffled in the HyperHdrX class
    def load_data(self, gt, batch_size, split=True, shuffle=True, bounding_box=None):
        self.hyperparams['bounding_box'] = bounding_box
        data_ = GeoHyperX(self.img_pth, gt, self.img_min, self.img_max, shuffle, **self.hyperparams)
        use_cuda = self.hyperparams['device'] == 'cuda'
        N = len(data_)

        if split:
            #split the indices into two continuous index arrays
            indices = np.arange(N)
            split_indice = int(0.95*N)
            train_indices, val_indices = indices[:split_indice], indices[split_indice:]

            #create Subset Sampler with previous index arrays
            train_sampler = SubsetSampler(train_indices)
            val_sampler = SubsetSampler(val_indices)

            #create DataLoader with previous Subset Sampler
            train_loader  = data.DataLoader(data_, sampler=train_sampler,
                                      batch_size=batch_size, pin_memory=use_cuda)
            val_loader  = data.DataLoader(data_, sampler = val_sampler,
                                      batch_size=batch_size, pin_memory=use_cuda)
            return train_loader, val_loader
        else:
            loader  = data.DataLoader(data_, batch_size=batch_size, pin_memory=use_cuda)
            return loader

    def subsample_loader(self, gt, split, batch_size, bounding_box=None):
        self.hyperparams['bounding_box'] = bounding_box
        data_ = GeoHyperX(self.img_pth, gt, self.img_min, self.img_max, True, **self.hyperparams)
        use_cuda = self.hyperparams['device'] == 'cuda'
        N = len(data_)
        
        #split the indices into two continuous index arrays
        indices = np.arange(N)
        
        split_indice = int(split*N)
        indices = indices[:split_indice]

        #create Subset Sampler with previous index arrays
        sampler = SubsetSampler(indices)

        #create DataLoader with previous Subset Sampler
        loader  = data.DataLoader(data_, sampler=sampler,
                                  batch_size=batch_size, pin_memory=use_cuda)

        return loader

    def pool_segmentation_(self, bounding_box, n_segments, compactness):
        data_src = rio.open(self.img_pth)
        data = data_src.read(self.rgb_bands, 
                             window=Window.from_slices(
                                (bounding_box[0][1], bounding_box[1][1]),
                                (bounding_box[0][0], bounding_box[1][0])
                                )
                             )
        data = data[:,:,:]
        data = data.transpose(1, 2, 0)
        data = (data - self.img_min) / (self.img_max - self.img_min)
        mask = self.pool.data[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]] != 0
        self.segmentation = slic(data, n_segments=int(n_segments), compactness=int(compactness), mask=mask)
        self.cluster_coordinates = {}
        for cluster_id in np.unique(self.segmentation):
            if cluster_id > 0:
                coords = np.where(cluster_id==self.segmentation)
                self.cluster_coordinates[cluster_id] = coords # boundin_box reference coordinates
                #tuple((coords[0]+bounding_box[0][1], coords[1]+bounding_box[0][0]))

        gt = rio.open(self.gt_pth)
        profile = gt.profile 
        with rasterio.open(self.segmentation_file, 'w', **profile) as dst:
            dst.write(self.segmentation.astype(rasterio.uint8), 1, 
                      window=Window.from_slices(
                            (bounding_box[0][1], bounding_box[1][1]),
                            (bounding_box[0][0], bounding_box[1][0])
                            ))

        # import matplotlib.pyplot as plt 
        # fig = plt.figure()
        # plt.imshow(self.segmentation)
        # plt.show()

        print("Bounding_box: ", bounding_box)
        print("cluster 1: ", self.cluster_coordinates[1])

    def superpixels_loader(self, bounding_box, batch_size=10):
        return SuperpixelsLoader(self, bounding_box, batch_size)


    def data(self, gt):
        mask = gt != 0
        return self.img[mask]

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
        return (self.train_gt.data + self.pool.data).flatten()

    @property
    def n_train(self):
        return self.train_gt.size

    @property
    def proportions(self):
        prop = np.zeros(self.n_classes-1)
        n = np.sum(self.train_gt.data != 0)
        for class_id in np.unique(self.train_gt.data):
            if class_id != 0:
                prop[class_id-1] = round(np.sum(self.train_gt.data == class_id) / n *100, 1)
        return prop
        

class SuperpixelsLoader:
    def __init__(self, dataset, bounding_box, batch_size):
        self.data_pth = dataset.img_pth 
        self.cluster_coordinates = dataset.cluster_coordinates 
        self.n_bands = dataset.n_bands
        self.bbl_index = dataset.bbl_index
        self.batch_size = batch_size
        self.data_src = rio.open(self.data_pth)
        self.dataset = dataset
        self.bounding_box = bounding_box
    
    def __len__(self):
        return len(self.cluster_coordinates)//self.batch_size

    def __iter__(self):
        return iter(self.generator())

    def generator(self):
        ROW_SIZE, NUM_COLUMNS = len(self.cluster_coordinates), self.n_bands
        for i in range(1, ROW_SIZE+1, self.batch_size):
            coordx, coordy = [], []
            spectra = torch.zeros((min(self.batch_size, ROW_SIZE-i), NUM_COLUMNS))

            for j in range(i, min(i+self.batch_size, ROW_SIZE)):
                coordx.extend(list(self.cluster_coordinates[j][0]))
                coordy.extend(list(self.cluster_coordinates[j][1]))
            min_row, max_row = min(coordx), max(coordx)
            min_col, max_col = min(coordy), max(coordy)
            # print(min_row, min_col)

            # print(i,min(i+self.batch_size, ROW_SIZE))

            data = self.data_src.read(self.bbl_index, 
                                window=Window.from_slices(
                                (min_row+self.bounding_box[0][1], max_row+self.bounding_box[0][1]+1),
                                (min_col+self.bounding_box[0][0], max_col+self.bounding_box[0][0]+1)
                                )
                             )
            data = (data - self.dataset.img_min) / (self.dataset.img_max - self.dataset.img_min)
            data = data.transpose(1,2,0)
            # print(data.shape)
            
            # import matplotlib.pyplot as plt 
            # A = np.zeros_like(self.dataset.segmentation)
            # A[min_row-self.bounding_box[0][1]:max_row-self.bounding_box[0][1]+1, min_col-self.bounding_box[0][0]:max_col-self.bounding_box[0][0]+1] = 1
            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(A)
            # ax[1].imshow(self.dataset.segmentation)
            # plt.show()

            # fig = plt.figure()
            # plt.imshow(data[:,:,50])
            # plt.show()

            coordx, coordy = np.zeros((min(self.batch_size, ROW_SIZE-i))), np.zeros((min(self.batch_size, ROW_SIZE-i)))
            for j in range(i, min(i+self.batch_size, ROW_SIZE)):
                cluster_id = j  
                coordinates = self.cluster_coordinates[cluster_id]
                random_ind = np.random.randint(len(coordinates[0])) 
                # print(coordinates[0][random_ind], coordinates[1][random_ind])

                coordx[j-i] = coordinates[0][random_ind]+self.bounding_box[0][1]
                coordy[j-i] = coordinates[1][random_ind]+self.bounding_box[0][0]

                coordinates = tuple((coordinates[0]-min_row, coordinates[1]-min_col))
                sp =  torch.from_numpy(np.mean(data[coordinates], axis=0).reshape(1,-1))
                spectra[j-i,:] = sp

                # import matplotlib.pyplot as plt 
                # fig, ax = plt.subplots(1,2)
                # A = np.zeros_like(self.dataset.segmentation)
                # A[self.cluster_coordinates[cluster_id]] = 1
                # ax[0].plot(spectra[j-i,:].reshape(-1))
                # ax[1].imshow(A)
                # plt.show()

            coords = tuple((torch.from_numpy(coordx.astype(np.int)), torch.from_numpy(coordy.astype(np.int))))
            yield spectra, None, coords


class SubsetSampler(data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)    


class GeoHyperX(torch.utils.data.Dataset):
    """ Generic class for a georeferenced hyperspectral dataset"""

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
        super(GeoHyperX, self).__init__()

        self.label = np.copy(gt.data)

        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.data_min = data_min
        self.data_max = data_max
        self.shuffle = shuffle
        bounding_box = hyperparams['bounding_box']

        self.coordinates = np.array(np.where(self.label == self.label)).T

        #open image data file
        self.data_src = rio.open(data_pth)
        
        #Get bad bands
        bbl = self.data_src.tags(ns=self.data_src.driver)['bbl'].replace(' ', '').replace('{', '').replace('}', '').split(',')
        bbl = np.array(list(map(int, bbl)), dtype=int)
        self.bbl_index = tuple(np.where(bbl != 0)[0] + 1)

        #create list of slices of the image based on block_hw
        block_hw = 4, 4
        bh = gt.data.shape[0] // block_hw[0]
        bw = gt.data.shape[1] // block_hw[1]
        self.blocks_slices = []
        for i in range(block_hw[0]):
            for j in range(block_hw[1]):
                slice_x = (i*bh, gt.data.shape[0]) if i == block_hw[0]-1 else (i*bh, (i+1)*bh)
                slice_y = (j*bw, gt.data.shape[1]) if j == block_hw[1]-1 else (j*bw, (j+1)*bw)
                self.blocks_slices.append((slice_x, slice_y))

        #remove ignored labels from gt
        mask = np.ones_like(gt.data)
        for l in self.ignored_labels:
            mask[gt.data == l] = 0

        #get all non zero pixel indice (row and col)

        x_pos, y_pos = np.nonzero(mask)
        if bounding_box is not None:
            in_box = (x_pos > bounding_box[0][1]) * (y_pos > bounding_box[0][0]) * (x_pos < bounding_box[1][1]) * (y_pos < bounding_box[1][0])
            x_pos, y_pos = x_pos[in_box], y_pos[in_box]


        #arrange non zero pixels indices by block 
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

        #remove possibly empty blocks
        i = 0
        while(i < len(self.indices)):
            if self.indices[i] == []:
                del self.blocks_slices[i]
                del self.indices[i]
            else:
                i+=1

        #Convert blocks slices and indices into numpy array
        self.blocks_slices = np.array(self.blocks_slices, dtype=tuple)
        self.indices = np.array(self.indices, dtype=list)


    def closeDataFile(self):
        self.data_src.close()

    def __len__(self):
        return sum([len(self.indices[i]) for i in range(len(self.indices))])

    def __getitem__(self, i):
        
        if i == 0:
            if self.shuffle:
                #shuffle pixels in each blocks and shuffle blocks
                for indice in self.indices:
                    np.random.shuffle(indice)
                random_idx = np.arange(len(self.indices))
                np.random.shuffle(random_idx)
                self.blocks_slices[random_idx]
                self.indices[random_idx]

            #init block index, block lenght and load fisrt block
            self.block_index = 0
            self.data_block = self.data_src.read(self.bbl_index, window=Window.from_slices(tuple(self.blocks_slices[self.block_index][0]), tuple(self.blocks_slices[self.block_index][1])))
            self.len_last = 0
            self.len_curr = len(self.indices[self.block_index])
        else:
            if i >= self.len_curr:
                #change block index, block lenght and load next block
                self.block_index += 1
                self.data_block = self.data_src.read(self.bbl_index, window=Window.from_slices(tuple(self.blocks_slices[self.block_index][0]), tuple(self.blocks_slices[self.block_index][1])))
                self.len_last = self.len_curr
                self.len_curr += len(self.indices[self.block_index])

        #get pixel coord based on i 
        x, y = self.indices[self.block_index][i-self.len_last]
        coord = [x,y]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        #get label
        label = self.label[x1:x2, y1:y2]

        #convert pixel coord to block index 
        x1, x2 = x1 - self.blocks_slices[self.block_index][0][0], x2 - self.blocks_slices[self.block_index][0][0]
        y1, y2 = y1 - self.blocks_slices[self.block_index][1][0], y2 - self.blocks_slices[self.block_index][1][0]

        #get data
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

        return data, label, coord

