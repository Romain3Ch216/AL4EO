import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import pickle as pkl
import numpy as np
from tqdm import tqdm
from learning.models import *
from learning.utils import *
import torch.utils.data
import os
import pickle as pkl
import gc
import time 

#===============================================================================
#         Classes that define query system for Active Learning
#===============================================================================

class Query:
    """
    Generic class to define a query system

    Args:
        n_px (int): number of pixels to query
        shuffle_prop (float): amount of noise to put in the ranking
        hyperparams (dict): hyperparameters
        reverse (bool): True if the most uncertain sample has the highest value
    """

    def __init__(self, n_px, hyperparams, shuffle_prop=0.0, reverse=False):
        self.n_px = n_px
        self.img_shape = hyperparams['img_shape']
        self.shuffle_prop = shuffle_prop
        self.reverse = reverse
        self.hyperparams = hyperparams
        self.score = None
        self.use_cuda = True if hyperparams['device'] == 'cuda' else False

    def train_loader(self, train_data):
        x_train, y_train = train_data
        train_dataset = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader  = data.DataLoader(train_dataset, shuffle=False,
                                  batch_size=self.hyperparams['batch_size'],
                                  pin_memory=self.use_cuda)
        return loader

    def compute_probs(self, model, data_loader):
        probs, coords =  model.predict_probs(data_loader, self.hyperparams)
        return probs, coords

    def compute_score(self):
        raise NotImplementedError

    def get_rank(self, score):
        ranks = np.argsort(score)
        if self.reverse:
            ranks = ranks[::-1]
        ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks.astype(int)

    def __call__(self, model, pool, train_data=None):
        self.score, self.coords = self.compute_score(model, pool)
        if isinstance(self.score, type(torch.zeros(1))):
            self.score = self.score.cpu().numpy()
        ranks = self.get_rank(self.score)
        selected = self.coords[ranks]
        selected = selected[:self.n_px]
        return selected

#===============================================================================
#                   Inter-class uncertainty heuristics
#===============================================================================

class BreakingTie(Query):
    """
    Class for Breaking Tie query system
    """

    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams, shuffle_prop, reverse=False)

    def compute_score(self, model, data_loader):
        probs, coords = self.compute_probs(model, data_loader)
        sorted_probs = np.sort(probs, axis=-1)
        breaking_ties = sorted_probs[:,-1] - sorted_probs[:,-2]
        return breaking_ties, coords

#===============================================================================
#                       Epsitemic uncertainty heuristics
#===============================================================================
#            Code from https://github.com/ElementAI/baal released under
#             the following Apache License 2.0 was partially modified
#===============================================================================
#                               Apache License
#                         Version 2.0, January 2004
#                      http://www.apache.org/licenses/

# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

# 1. Definitions.

#    "License" shall mean the terms and conditions for use, reproduction,
#    and distribution as defined by Sections 1 through 9 of this document.

#    "Licensor" shall mean the copyright owner or entity authorized by
#    the copyright owner that is granting the License.

#    "Legal Entity" shall mean the union of the acting entity and all
#    other entities that control, are controlled by, or are under common
#    control with that entity. For the purposes of this definition,
#    "control" means (i) the power, direct or indirect, to cause the
#    direction or management of such entity, whether by contract or
#    otherwise, or (ii) ownership of fifty percent (50%) or more of the
#    outstanding shares, or (iii) beneficial ownership of such entity.

#    "You" (or "Your") shall mean an individual or Legal Entity
#    exercising permissions granted by this License.

#    "Source" form shall mean the preferred form for making modifications,
#    including but not limited to software source code, documentation
#    source, and configuration files.

#    "Object" form shall mean any form resulting from mechanical
#    transformation or translation of a Source form, including but
#    not limited to compiled object code, generated documentation,
#    and conversions to other media types.

#    "Work" shall mean the work of authorship, whether in Source or
#    Object form, made available under the License, as indicated by a
#    copyright notice that is included in or attached to the work
#    (an example is provided in the Appendix below).

#    "Derivative Works" shall mean any work, whether in Source or Object
#    form, that is based on (or derived from) the Work and for which the
#    editorial revisions, annotations, elaborations, or other modifications
#    represent, as a whole, an original work of authorship. For the purposes
#    of this License, Derivative Works shall not include works that remain
#    separable from, or merely link (or bind by name) to the interfaces of,
#    the Work and Derivative Works thereof.

#    "Contribution" shall mean any work of authorship, including
#    the original version of the Work and any modifications or additions
#    to that Work or Derivative Works thereof, that is intentionally
#    submitted to Licensor for inclusion in the Work by the copyright owner
#    or by an individual or Legal Entity authorized to submit on behalf of
#    the copyright owner. For the purposes of this definition, "submitted"
#    means any form of electronic, verbal, or written communication sent
#    to the Licensor or its representatives, including but not limited to
#    communication on electronic mailing lists, source code control systems,
#    and issue tracking systems that are managed by, or on behalf of, the
#    Licensor for the purpose of discussing and improving the Work, but
#    excluding communication that is conspicuously marked or otherwise
#    designated in writing by the copyright owner as "Not a Contribution."

#    "Contributor" shall mean Licensor and any individual or Legal Entity
#    on behalf of whom a Contribution has been received by Licensor and
#    subsequently incorporated within the Work.

# 2. Grant of Copyright License. Subject to the terms and conditions of
#    this License, each Contributor hereby grants to You a perpetual,
#    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#    copyright license to reproduce, prepare Derivative Works of,
#    publicly display, publicly perform, sublicense, and distribute the
#    Work and such Derivative Works in Source or Object form.

# 3. Grant of Patent License. Subject to the terms and conditions of
#    this License, each Contributor hereby grants to You a perpetual,
#    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#    (except as stated in this section) patent license to make, have made,
#    use, offer to sell, sell, import, and otherwise transfer the Work,
#    where such license applies only to those patent claims licensable
#    by such Contributor that are necessarily infringed by their
#    Contribution(s) alone or by combination of their Contribution(s)
#    with the Work to which such Contribution(s) was submitted. If You
#    institute patent litigation against any entity (including a
#    cross-claim or counterclaim in a lawsuit) alleging that the Work
#    or a Contribution incorporated within the Work constitutes direct
#    or contributory patent infringement, then any patent licenses
#    granted to You under this License for that Work shall terminate
#    as of the date such litigation is filed.

# 4. Redistribution. You may reproduce and distribute copies of the
#    Work or Derivative Works thereof in any medium, with or without
#    modifications, and in Source or Object form, provided that You
#    meet the following conditions:

#    (a) You must give any other recipients of the Work or
#        Derivative Works a copy of this License; and

#    (b) You must cause any modified files to carry prominent notices
#        stating that You changed the files; and

#    (c) You must retain, in the Source form of any Derivative Works
#        that You distribute, all copyright, patent, trademark, and
#        attribution notices from the Source form of the Work,
#        excluding those notices that do not pertain to any part of
#        the Derivative Works; and

#    (d) If the Work includes a "NOTICE" text file as part of its
#        distribution, then any Derivative Works that You distribute must
#        include a readable copy of the attribution notices contained
#        within such NOTICE file, excluding those notices that do not
#        pertain to any part of the Derivative Works, in at least one
#        of the following places: within a NOTICE text file distributed
#        as part of the Derivative Works; within the Source form or
#        documentation, if provided along with the Derivative Works; or,
#        within a display generated by the Derivative Works, if and
#        wherever such third-party notices normally appear. The contents
#        of the NOTICE file are for informational purposes only and
#        do not modify the License. You may add Your own attribution
#        notices within Derivative Works that You distribute, alongside
#        or as an addendum to the NOTICE text from the Work, provided
#        that such additional attribution notices cannot be construed
#        as modifying the License.

#    You may add Your own copyright statement to Your modifications and
#    may provide additional or different license terms and conditions
#    for use, reproduction, or distribution of Your modifications, or
#    for any such Derivative Works as a whole, provided Your use,
#    reproduction, and distribution of the Work otherwise complies with
#    the conditions stated in this License.

# 5. Submission of Contributions. Unless You explicitly state otherwise,
#    any Contribution intentionally submitted for inclusion in the Work
#    by You to the Licensor shall be under the terms and conditions of
#    this License, without any additional terms or conditions.
#    Notwithstanding the above, nothing herein shall supersede or modify
#    the terms of any separate license agreement you may have executed
#    with Licensor regarding such Contributions.

# 6. Trademarks. This License does not grant permission to use the trade
#    names, trademarks, service marks, or product names of the Licensor,
#    except as required for reasonable and customary use in describing the
#    origin of the Work and reproducing the content of the NOTICE file.

# 7. Disclaimer of Warranty. Unless required by applicable law or
#    agreed to in writing, Licensor provides the Work (and each
#    Contributor provides its Contributions) on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#    implied, including, without limitation, any warranties or conditions
#    of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#    PARTICULAR PURPOSE. You are solely responsible for determining the
#    appropriateness of using or redistributing the Work and assume any
#    risks associated with Your exercise of permissions under this License.

# 8. Limitation of Liability. In no event and under no legal theory,
#    whether in tort (including negligence), contract, or otherwise,
#    unless required by applicable law (such as deliberate and grossly
#    negligent acts) or agreed to in writing, shall any Contributor be
#    liable to You for damages, including any direct, indirect, special,
#    incidental, or consequential damages of any character arising as a
#    result of this License or out of the use or inability to use the
#    Work (including but not limited to damages for loss of goodwill,
#    work stoppage, computer failure or malfunction, or any and all
#    other commercial damages or losses), even if such Contributor
#    has been advised of the possibility of such damages.

# 9. Accepting Warranty or Additional Liability. While redistributing
#    the Work or Derivative Works thereof, You may choose to offer,
#    and charge a fee for, acceptance of support, warranty, indemnity,
#    or other liability obligations and/or rights consistent with this
#    License. However, in accepting such obligations, You may act only
#    on Your own behalf and on Your sole responsibility, not on behalf
#    of any other Contributor, and only if You agree to indemnify,
#    defend, and hold each Contributor harmless for any liability
#    incurred by, or claims asserted against, such Contributor by reason
#    of your accepting any such warranty or additional liability.

# END OF TERMS AND CONDITIONS

# APPENDIX: How to apply the Apache License to your work.

#    To apply the Apache License to your work, attach the following
#    boilerplate notice, with the fields enclosed by brackets "[]"
#    replaced with your own identifying information. (Don't include
#    the brackets!)  The text should be enclosed in the appropriate
#    comment syntax for the file format. We also recommend that a
#    file or class name and description of purpose be included on the
#    same "printed page" as the copyright notice for easier
#    identification within third-party archives.

# Copyright [yyyy] [name of copyright owner]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==============================================================================

class BALD(Query):
    """
    Class for BALD query system

    Args:

    References:
        https://arxiv.org/abs/1703.02910
    """

    def __init__(self, n_px, hyperparams, shuffle_prop=0.0):
        super().__init__(n_px, hyperparams,
            shuffle_prop=shuffle_prop, reverse=True
        )

        self.device = hyperparams['device']

    def compute_score(self, model, data_loader):
        """
        Args:
            probs (ndarray): Array of predictions

        Returns:
            Array of scores.
        """
        probs, coords = self.compute_probs(model, data_loader)
        bald_acq = self.compute_score_from_probs(probs)
        return bald_acq, coords

    def compute_score_from_probs(self, probs):
        """
        Args:
            probs (ndarray): Array of probs

        Returns:
            Array of scores.
        """
        assert probs.ndim >= 3
        # [n_sample, n_class, ..., n_iterations]
        expected_entropy = - torch.mean(torch.sum(probs * torch.log(probs + 1e-5), 1),
                                        dim=-1)  # [batch size, ...]
        expected_p = torch.mean(probs, dim=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + 1e-5),
                                         dim=1)  # [batch size, ...]
        bald_acq = entropy_expected_p - expected_entropy
        return bald_acq


#===============================================================================
#                          Core-set Active Learning
#                          Parts of code were taken from
#           https://github.com/dsgissin/DiscriminativeActiveLearning
#                  released under the following MIT license
#===============================================================================
# MIT License

# Copyright (c) 2018 dsgissin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#===============================================================================

class Coreset(Query):
    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams,
            shuffle_prop=shuffle_prop, reverse=False
        )
        self.outlier_prop = hyperparams['outlier_prop']
        self.subsample = hyperparams['subsample']
        self.model_feature = 'cnn'
        self.hyperparams = hyperparams

    def greedy_k_center(self, model, labeled_features, unlabeled_pool, amount):
        t = time.time()
        greedy_indices = []
        min_dist = []
        coords_x, coords_y = [], []

        unlabeled_batch, _, unlabeled_coord = next(iter(unlabeled_pool))
        unlabeled_features = model.predict_batch(unlabeled_batch, self.hyperparams)

        dist = torch.cdist(labeled_features, unlabeled_features)
        min_dist, _ = torch.min(dist, dim=0)
        min_dist = min_dist.view(1, unlabeled_batch.shape[0])
        coords_x.extend(unlabeled_coord[0].numpy())
        coords_y.extend(unlabeled_coord[1].numpy())

        for i, (unlabeled_batch, _, unlabeled_coord) in enumerate(unlabeled_pool):
            if i > 0:
                unlabeled_features = model.predict_batch(unlabeled_batch, self.hyperparams)
                dist = torch.cdist(labeled_features, unlabeled_features)
                min_, _ = torch.min(dist, dim=0)
                min_ = min_.view(1, unlabeled_batch.shape[0])
                min_dist = torch.cat((min_dist, min_), dim=1)

                coords_x.extend(unlabeled_coord[0].numpy())
                coords_y.extend(unlabeled_coord[1].numpy())

        coords = np.array(tuple((coords_x, coords_y))).T
        farthest = torch.argmax(min_dist).item()
        greedy_indices.append(farthest)

        for i in range(amount-1):
            last_feature = model.predict_batch(self.get_sp_from_img(coords[greedy_indices[-1]]), self.hyperparams)
            dist_ = []
            for unlabeled_batch, _, unlabeled_coord in unlabeled_pool:
                unlabeled_features = model.predict_batch(unlabeled_batch, self.hyperparams)
                dist = torch.cdist(last_feature, unlabeled_features)
                dist_.append(dist)

            dist_ = torch.cat(dist_, dim=1)
            min_dist = torch.cat((min_dist, dist_), dim=0)
            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.view(1, min_dist.shape[0])
            farthest = torch.argmax(min_dist).item()
            greedy_indices.append(farthest)

        max_dist = torch.max(min_dist) 
        print("Finished computing greedy solution in {:.1f}s!".format(time.time()-t))

        coords_x = np.array(coords_x)
        coords_y = np.array(coords_y)
        coords_x, coords_y = coords_x[greedy_indices], coords_y[greedy_indices]
        coordinates = np.array(tuple((coords_x, coords_y))).T
        return coordinates, max_dist

    def get_sp_from_img(self, coord):
        import rasterio
        import rasterio as rio
        from rasterio.windows import Window

        data = rio.open(self.hyperparams['img_pth'])
        n_bands = data.count
        if 'bbl' in data.tags(ns=data.driver):
            bbl = data.tags(ns=data.driver)['bbl'].replace(' ', '').replace('{', '').replace('}', '').split(',')
            bbl = np.array(list(map(int, bbl)), dtype=int)
        else:
            bbl = np.ones(n_bands, dtype=np.int)
        bbl_index = tuple(np.where(bbl != 0)[0] + 1)

        img_min, img_max, _ = rasterio.rio.insp.stats(data.read(bbl_index))
        sp = data.read(bbl_index, window=Window(coord[1], coord[0], 1, 1)).reshape(1, -1)
        sp = (sp-img_min)/(img_max-img_min)
        return torch.from_numpy(sp).float()

    # def get_neighborhood_graph(self, model, labeled_pool, unlabeled_pool, delta):
    #     t = time.time()
    #     graph = {
    #         'nodes':[], 
    #         'neighbors': [], 
    #         'distances': [], 
    #         'maximum': 0, 
    #         'minimum': np.inf
    #         }
    #     i = 0
    #     for batch, _, _ in concatenate_loaders(labeled_pool, unlabeled_pool):
    #         features_1 = model.predict_batch(batch, self.hyperparams)
    #         dist = []
    #         for j, (batch_2, _, _) in enumerate(concatenate_loaders(labeled_pool, unlabeled_pool)):
    #             features_2 = model.predict_batch(batch_2, self.hyperparams)
    #             dist.append(torch.cdist(features_1, features_2))

    #         dist = torch.cat(dist, dim=1)
    #         mask = dist <= delta 
    #         indices = np.where(mask)
    #         distances = dist[mask]
    #         graph['nodes'].extend(indices[0])
    #         graph['neighbors'].extend(indices[1])
    #         graph['distances'].extend(distances)
    #         graph['maximum'] = max(graph['maximum'], torch.max(distances).item())
    #         graph['minimum'] = min(graph['minimum'], torch.min(distances).item())
            
    #     print("Finished Building Graph in {:.1f}s!".format(time.time()-t))
    #     return graph

    def __call__(self, model, unlabeled_pool, labeled_pool):
        n_labeled = get_size_loader(labeled_pool)
        labeled_idx = np.arange(n_labeled)
        n_unlabeled = get_size_loader(unlabeled_pool)
        unlabeled_idx = np.arange(n_labeled, n_labeled + n_unlabeled)
        outlier_count = int((n_labeled+n_unlabeled) * self.outlier_prop)
        n_total = n_labeled + n_unlabeled

        train_representation, _ = self.compute_probs(model, labeled_pool)
        # use the learned representation for the k-greedy-center algorithm:
        print("Calculating Greedy K-Center Solution...")
        greedy_solution, max_delta = self.greedy_k_center(model, train_representation, unlabeled_pool, self.n_px)
        return greedy_solution



#===============================================================================
#                                  Utils
#===============================================================================
def _shuffle_subset(data: np.ndarray, shuffle_prop: float) -> np.ndarray:
    to_shuffle = np.nonzero(np.random.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[np.random.permutation(to_shuffle), ...]
    return data

#===============================================================================
#                     Loading function
#===============================================================================
def load_query(config, dataset):
    device = config.setdefault('device', torch.device('cpu'))
    n_classes = config['n_classes']
    n_bands = config['n_bands']
    epochs = config.setdefault('epochs', 100)
    config.setdefault('center_pixel', True)
    config.setdefault('patch_size', 1)
    config.setdefault('n_px', 5)
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(config['ignored_labels'])] = 0.
    weights = weights.to(config['device'])
    weights = config.setdefault('weights', weights)

    if config['query'] == 'bald':
        from learning.consistent_dropout import MCConsistentDropoutModule
        config.setdefault('batch_size', 128)
        lr = config.setdefault('learning_rate', 0.01)
        config.setdefault('num_samples', 30)
        net = HuEtAl(dataset.n_bands, dataset.n_classes, dropout=0.5)
        net = MCConsistentDropoutModule(net)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=config['weight_decay'])
        config['optimizer'] = optimizer
        criterion = nn.CrossEntropyLoss(weight=config['weights'])
        config.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=config['epochs']//4, verbose=True))
        model = BayesianModelWrapper(net, criterion)
        query = BALD(config['n_px'], config, shuffle_prop=0)

    elif config['query'] == 'breaking_tie':
        config.setdefault('batch_size', 128)
        config.setdefault('weight_decay', 0)
        lr = config.setdefault('learning_rate', 0.01)
        net = HuEtAl(dataset.n_bands, dataset.n_classes)
        net = net.to(config['device'])
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss(weight=config['weights'])
        config.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=config['epochs']//4, verbose=True))
        model = NeuralNetwork(net, optimizer, criterion)
        query = BreakingTie(config['n_px'], config, shuffle_prop=0)

    elif config['query'] == 'coreset':
        config.setdefault('batch_size', 128)
        config.setdefault('weight_decay', 0)
        config.setdefault('outlier_prop', 1e-4)
        lr = config.setdefault('learning_rate', 0.01)
        net = HuEtAl(dataset.n_bands, dataset.n_classes)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss(weight=config['weights'])
        config.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=config['epochs']//4, verbose=True))
        model = NeuralNetwork(net, optimizer, criterion)
        query = Coreset(config['n_px'], config, shuffle_prop=0)

    return model, query, config
