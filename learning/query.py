import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import pickle as pkl
import numpy as np
from scipy.special import xlogy
from tqdm import tqdm
from learning.models import *
from scipy.spatial import distance_matrix
from ortools.linear_solver import pywraplp
from learning.utils import *
import torch.utils.data
import os
import pickle as pkl
import gc

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


class RandomSampling(Query):
    """
    Class for Breaking Tie query system
    """

    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams, shuffle_prop, reverse=False)

    def compute_score(self, model, pool):
        x_pool, y_pool = pool
        score = np.random.rand(x_pool.shape[0])
        return score

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


class MaxEntropy(Query):
    """
    Class for Max Entropy query system
    A tester
    """

    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams,
            shuffle_prop=shuffle_prop, reverse=True)

        self.softmax = nn.Softmax(dim=-1)

    def compute_score(self, model, pool):
        data_loader = self.pool_loader(pool)
        probs = self.compute_probs(model, data_loader)
        probs = self.softmax(probs)
        log_probs = torch.log(probs)
        entropy = - torch.sum(probs*log_probs, dim=-1)
        return entropy


class VariationRatios(Query):
    """
    Class for Variation Ratios query system
    A tester
    """

    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams,
            shuffle_prop=shuffle_prop, reverse=True)

        self.softmax = nn.Softmax(dim=-1)

    def compute_score(self, model, pool):
        data_loader = self.pool_loader(pool)
        probs = self.compute_probs(model, data_loader)
        probs = self.softmax(probs)
        max_probs, _ = torch.max(probs, dim=-1)
        ratios = 1 - max_probs
        return ratios

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

class BatchBALD(BALD):
    """
    Implementation of BatchBALD from https://github.com/BlackHC/BatchBALD

    Args:
        num_samples (int): Number of samples to select.
        num_draw (int): Number of draw to perform from the history.
                        From the paper `40000 // num_classes` is suggested.
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none').

    Notes:
        This implementation only returns the ranking and not the score.

    References:
        https://arxiv.org/abs/1906.08158

    Notes:
        K = iterations, C=classes
        Not tested on 4+ dims.
        """

    def __init__(self, n_px, hyperparams, shuffle_prop=0.0):
        self.epsilon = 1e-5
        self.num_samples = hyperparams['num_samples']
        self.num_draw = hyperparams['num_draw']
        self.subsample = hyperparams['subsample']

        super().__init__(n_px, hyperparams, shuffle_prop=0.0)

    def _draw_choices(self, probs, n_choices):
        """
        Draw `n_choices` sample from `probs`.

        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/torch_utils.py#L187

        Returns:
            choices: B... x `n_choices`

        """
        probs = probs.permute(0, 2, 1)
        probs_B_C = probs.reshape((-1, probs.shape[-1]))

        # samples: Ni... x draw_per_xx
        choices = torch.multinomial(probs_B_C,
                                    num_samples=n_choices, replacement=True)

        choices_b_M = choices.reshape(list(probs.shape[:-1]) + [n_choices])
        return choices_b_M.long()

    def _sample_from_history(self, probs, num_draw=100):
        """
        Sample `num_draw` choices from `probs`

        Args:
            probs (Tensor[batch, classes, ..., iterations]): Tensor to be sampled from.
            num_draw (int): Number of draw.

        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/joint_entropy/sampling.py

        Returns:
            Tensor[num_draw, iterations]
        """
        # probs = torch.from_numpy(probs).double()
        # pdb.set_trace()

        n_iterations = probs.shape[-1]

        # [batch, draw, iterations]
        choices = self._draw_choices(probs, num_draw)

        # [batch, iterations, iterations, draw]
        expanded_choices_N_K_K_S = choices[:, None, :, :]
        expanded_probs_N_K_K_C = probs.permute(0, 2, 1)[:, :, None, :]


        probs = gather_expand(expanded_probs_N_K_K_C, dim=-1, index=expanded_choices_N_K_K_S)

        # exp sum log seems necessary to avoid 0s?
        entropies = torch.exp(torch.sum(torch.log(probs), dim=0, keepdim=False))
        entropies = entropies.reshape((n_iterations, -1))

        samples_M_K = entropies.t()
        return samples_M_K

    def _conditional_entropy(self, probs):
        K = probs.shape[-1]
        return torch.sum(-probs * torch.log(probs + 1e-5), dim=(1,-1)) / K

    def _joint_entropy(self, predictions, selected):
        """
        Compute the joint entropy between `preditions` and `selected`
        Args:
            predictions (Tensor): First tensor with shape [B, C, Iterations]
            selected (Tensor): Second tensor with shape [M, Iterations].

        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/joint_entropy/sampling.py

        Notes:
            Only Classification is supported, not semantic segmentation or other.

        Returns:
            Generator yield B entropies.
        """
        K = predictions.shape[-1]
        C = predictions.shape[1]
        B = predictions.shape[0]
        M = selected.shape[0]
        predictions = predictions.transpose(1, 2)
        exp_y = torch.cat(
            [torch.matmul(selected, predictions[i]).unsqueeze(0) for i in range(predictions.shape[0])], dim=0) / K

        assert exp_y.shape == (B, M, C)
        mean_entropy = torch.mean(selected, dim=-1, keepdim=True)[None]
        assert mean_entropy.shape == (1, M, 1)

        step = 256
        for idx in range(0, exp_y.shape[0], step):
            b_preds = exp_y[idx:idx + step]
            yield torch.sum(-b_preds*torch.log(b_preds + 1e-5) / mean_entropy, dim=(1, -1)).cpu().numpy() / M



    def compute_score(self, model, data_loader):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions [batch_size, C, Iterations]

        Notes:
            Only Classification is supported, not semantic segmentation or other.

        Returns:
            Array of scores.
        """
        probs = self.compute_probs(model, data_loader)
        if torch.cuda.is_available():
            print("Running on cuda...")
            probs = probs.to('cuda')

        if self.subsample:
            num_subsamples = int(self.subsample * probs.shape[0])
            indices = np.arange(probs.shape[0])
            subsample_idx = np.random.choice(indices, num_subsamples, replace=False)
            probs = probs[subsample_idx]

        MIN_SPREAD = 0.1
        COUNT = 0
        # Get conditional_entropies_B
        print("Get conditional_entropies_B...")
        conditional_entropies_B = self._conditional_entropy(probs) # [N]
        print("Add px 1...")
        bald_out = super().compute_score_from_probs(probs)
        self.score = bald_out

        # We start with the most uncertain sample according to BALD.
        history = bald_out.argsort()[-1:].tolist() # [1]

        for step in range(1, self.n_px):
            print("Add px {}...".format(step+1))
            # Draw `num_draw` example from history, take entropy
            # TODO use numpy/numba

            #probs[history] : [len(history) x n_classes x iterations]
            selected = self._sample_from_history(probs[history], num_draw=self.num_draw)

            # Compute join entropy
            joint_entropy = list(self._joint_entropy(probs, selected))
            joint_entropy = np.concatenate(joint_entropy)

            partial_multi_bald_b = joint_entropy - conditional_entropies_B.cpu().numpy()
            partial_multi_bald_b = partial_multi_bald_b
            partial_multi_bald_b[..., np.array(history)] = 0
            # Add best to history
            partial_multi_bald_b = partial_multi_bald_b.squeeze()
            assert partial_multi_bald_b.ndim == 1
            winner_index = partial_multi_bald_b.argmax()
            history.append(winner_index)

            if partial_multi_bald_b.max() < MIN_SPREAD:
                COUNT += 1
                if COUNT > 50 or len(history) >= probs.shape[0]:
                    break
        coordinates = np.array(history)
        if self.subsample:
            coordinates = subsample_idx[coordinates]
        return coordinates


    def __call__(self, model, pool, train_data):
        ranks = self.compute_score(model, pool)
        return ranks


class Disagreement(Query):
    """
    Class for Disagreement query system
    A tester
    """

    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams,
            shuffle_prop=shuffle_prop, reverse=True)

    def compute_score(self, model, pool):
        data_loader = self.pool_loader(pool)
        probs = self.compute_probs(model, data_loader)
        preds = np.argmax(probs, axis=1)
        disagreement = np.zeros(preds.shape[0])

        for i in range(preds.shape[0]):
            disagreement[i] = len(np.unique(preds[i]))

        return disagreement


class Variance(Query):
    """
    Class for Variance query system
    """

    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams,
            shuffle_prop=shuffle_prop, reverse=True)
        self.softmax = nn.Softmax(dim=-1)

    def compute_score(self, model, pool):
        data_loader = self.pool_loader(pool)
        probs = self.compute_probs(model, data_loader)
        probs = self.softmax(probs)

        P = probs.unsqueeze(-1)
        mean_oueter_product = torch.mean(torch.matmul(P, P.transpose(-1,-2)), dim=1)

        mean = torch.mean(probs, dim=1)
        mean = mean.unsqueeze(-1)

        outer_product_mean = torch.matmul(mean, mean.transpose(-1,-2))

        var = mean_oueter_product - outer_product_mean
        var = np.trace(var, axis1=1, axis2=2)

        return var

#===============================================================================
#                    Adversarial active learning heuristic
#===============================================================================

class AdversarialSampler(Query):
    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams,
            shuffle_prop=shuffle_prop, reverse=False
        )

    def compute_score(self, model, pool):
        all_preds = []
        all_indices = []

        batch_size = self.hyperparams['batch_size']
        x_pool, y_pool = pool
        score = np.zeros(x_pool.shape[0])
        data_loader = self.pool_loader(pool)
        for batch_id, (data, _) in enumerate(data_loader):
            data = data.to(self.hyperparams['device'])
            model.vae.to(self.hyperparams['device'])
            model.discriminator.to(self.hyperparams['device'])

            with torch.no_grad():
                _, _, mu, _ = model.vae(data)
                score[batch_id*batch_size:min(x_pool.shape[0], (batch_id+1)*batch_size)] = model.discriminator(mu).cpu()

        return score

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

    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            if i%1000==0:
                print("At Point " + str(i))
            # Calcule la distance avec les greedy points et pas avec la base d'apprentissage ? OK
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices, dtype=int), np.max(min_dist)

    def get_distance_matrix(self, X, Y):
        dist_mat = np.matmul(X,Y.transpose())
        x_norm = np.linalg.norm(X, axis=-1).reshape(-1,1)
        y_norm = np.linalg.norm(Y, axis=-1).reshape(1,-1)
        dist_mat = x_norm**2 + y_norm**2 - 2*dist_mat
        dist_mat = np.clip(dist_mat, 0, 10000)
        dist_mat = np.sqrt(dist_mat)
        return dist_mat

    def get_neighborhood_graph(self, representation, delta):

        graph = {}
        print(representation.shape)
        for i in range(0, representation.shape[0], 1000):

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
                amount = representation.shape[0] - i
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i+amount):
                graph[j] = [(idx, distances[j-i, idx]) for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]

        print("Finished Building Graph!")
        return graph

    def get_graph_max(self, representation, delta):

        print("Getting Graph Maximum...")

        maximum = 0
        for i in range(0, representation.shape[0], 1000):
            print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)

            distances = np.reshape(distances, (-1))
            distances[distances > delta] = 0
            maximum = max(maximum, np.max(distances))

        return maximum

    def get_graph_min(self, representation, delta):

        print("Getting Graph Minimum...")

        minimum = 10000
        for i in range(0, representation.shape[0], 1000):
            print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)

            distances = np.reshape(distances, (-1))
            distances[distances < delta] = 10000
            minimum = min(minimum, np.min(distances))

        return minimum

    def mip_model(self, representation, labeled_idx, budget, delta, outlier_count, greedy_indices=None):

        model = pywraplp.Solver.CreateSolver('SCIP')

        # set up the variables:
        points = {}
        outliers = {}
        for i in range(representation.shape[0]):
            if i in labeled_idx:
                points[i] = model.NumVar(1.0, 1.0, name="points_{}".format(i))
            else:
                points[i] = model.BoolVar(name="points_{}".format(i))

            outliers[i] = model.BoolVar(name="outliers_{}".format(i))
            outliers[i].start = 0

        # initialize the solution to be the greedy solution:
        if greedy_indices is not None:
            for i in greedy_indices:
                points[i].start = 1.0

        # set the outlier budget:
        model.Add(sum(outliers[i] for i in outliers) <= outlier_count, "budget")

        # build the graph and set the constraints:
        model.Add(sum(points[i] for i in range(representation.shape[0])) == budget, "budget")
        neighbors = {}
        graph = {}
        print("Updating Neighborhoods In MIP Model...")
        for i in range(0, representation.shape[0], 1000):
            print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
                amount = representation.shape[0] - i
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i+amount):
                graph[j] = [(idx, distances[j-i, idx]) for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors[j] = [points[idx] for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors[j].append(outliers[j])
                model.Add(sum(neighbors[j]) >= 1, "coverage+outliers")

        return model, graph, points, outliers


    def __call__(self, model, pool, train_data):
        train_representation, _ = train_data
        labeled_idx = np.arange(train_representation.shape[0])
        pool_representation, _ =  pool
        unlabeled_idx = np.arange(train_representation.shape[0], train_representation.shape[0] + pool_representation.shape[0])
        representation = np.concatenate((train_representation, pool_representation), axis=0)
        outlier_count = int(representation.shape[0] * self.outlier_prop)

        start_time = time.time()

        train_loader = self.train_loader(train_data)
        train_representation = self.compute_probs(model, train_loader)
        pool_loader = self.pool_loader(pool)
        pool_representation = self.compute_probs(model, pool_loader)

        if self.subsample:
            subsample_num = int(self.subsample*pool_representation.shape[0])
            print('Subsampling... {} samples'.format(subsample_num))
            subsample_idx = np.random.choice(np.arange(pool_representation.shape[0]), subsample_num, replace=False)
            pool_representation = pool_representation[subsample_idx]
            unlabeled_idx = np.arange(train_representation.shape[0], train_representation.shape[0] + pool_representation.shape[0])

        representation = np.concatenate((train_representation, pool_representation), axis=0)
        print('Total number of training samples: {}'.format(train_representation.shape[0]))
        print('Total number of pool samples: {}'.format(pool_representation.shape[0]))

        t = time.time() - start_time
        print('Feature extraction done in {}m...'.format(t/60))

        # use the learned representation for the k-greedy-center algorithm:
        print("Calculating Greedy K-Center Solution...")
        greedy_solution, max_delta = self.greedy_k_center(train_representation, pool_representation, self.n_px)
        new_indices = unlabeled_idx[greedy_solution]
        # submipnodes = 20000

        # iteratively solve the MIP optimization problem:
        eps = 0.01
        upper_bound = max_delta
        lower_bound = max_delta / 2.0
        print("Building MIP Model...")
        model, graph, points, outliers =\
         self.mip_model(representation, labeled_idx, len(labeled_idx) + self.n_px, upper_bound, outlier_count, greedy_indices=new_indices)
        # model.Params.SubMIPNodes = submipnodes
        model.Solve()
        indices = [i for i in graph if points[i].solution_value() == 1]
        current_delta = upper_bound
        while upper_bound - lower_bound > eps:

            print("upper bound is {ub}, lower bound is {lb}".format(ub=upper_bound, lb=lower_bound))
            if model.Solve() != pywraplp.Solver.OPTIMAL:
                print("Optimization Failed - Infeasible!")

                lower_bound = max(current_delta, self.get_graph_min(representation, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0

                del model
                gc.collect()
                model, graph, points, outliers =\
                    self.mip_model(representation, labeled_idx, len(labeled_idx) + self.n_px, current_delta, outlier_count, greedy_indices=indices)
                # model.Params.SubMIPNodes = submipnodes

            else:
                print("Optimization Succeeded!")
                upper_bound = min(current_delta, self.get_graph_max(representation, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0
                indices = [i for i in graph if points[i].solution_value() == 1]

                del model
                gc.collect()
                model, graph, points, outliers = self.mip_model(representation, labeled_idx, len(labeled_idx) + self.n_px, current_delta, outlier_count, greedy_indices=indices)
                # model.Params.SubMIPNodes = submipnodes

            if upper_bound - lower_bound > eps:
                model.Solve()

        if len(indices)>0:
        #if len(indices) == self.n_px:
            indices = np.array(indices)
        else:
            print('Using greedy solution...')
            indices = greedy_solution

        indices = indices[indices >= train_representation.shape[0]]
        indices = indices - train_representation.shape[0]
        if self.subsample:
            indices = subsample_idx[indices]
        return indices



#===============================================================================
#                             Cluster based AL
#===============================================================================
#                                Code from
# https://github.com/google/active-learning/blob/master/sampling_methods/hierarchical_clustering_AL.py
#             released under the following license was partially modified
#===============================================================================
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import time


class HierarchicalClusterAL(Query):
  """Implements hierarchical cluster AL based method.
  All methods are internal.  select_batch_ is called via abstract classes
  outward facing method select_batch.
  Default affininity is euclidean and default linkage is ward which links
  cluster based on variance reduction.  Hence, good results depend on
  having normalized and standardized data.
  """
  def __init__(self, n_px, hyperparams, shuffle_prop, dataset, seed, \
               affinity='euclidean', linkage='ward', clustering=None,\
               max_features=2):
    super().__init__(n_px, hyperparams, shuffle_prop=shuffle_prop, reverse=False)

    """Initializes AL method and fits hierarchical cluster to data.
    Args:
      X: data
      y: labels for determinining number of clusters as an input to
        AgglomerativeClustering
      seed: random seed used for sampling datapoints for batch
      beta: width of error used to decide admissble labels, higher value of beta
        corresponds to wider confidence and less stringent definition of
        admissibility
        See scikit Aggloerative clustering method for more info
      affinity: distance metric used for hierarchical clustering
      linkage: linkage method used to determine when to join clusters
      clustering: can provide an AgglomerativeClustering that is already fit
      max_features: limit number of features used to construct hierarchical
        cluster.  If specified, PCA is used to perform feature reduction and
        the hierarchical clustering is performed using transformed features.
    """
    self.name = 'hierarchical'
    self.subsample = hyperparams['subsample']
    self.max_features = max_features
    self.affinity = affinity
    self.linkage = linkage
    self.seed = seed
    np.random.seed(seed)
    # Variables for the hierarchical cluster
    if clustering is not None:
      self.model = clustering
      self.already_clustered = True
    self.beta = hyperparams['beta']
    self.init_at_each_step = False

    if hyperparams['superpixels'] or hyperparams['subsample']:
        self.init_at_each_step = True
    else:
        self.initialize_hierarchy(dataset.train_data, dataset.pool_data)


  def initialize_hierarchy(self, train_data, pool_data):
    self.n_leaves = None
    self.n_components = None
    self.children_list = None
    self.node_dict = None
    self.root = None  # Node name, all node instances access through self.tree
    self.tree = None
    self.labels = {}
    self.pruning = []
    self.admissible = {}
    self.selected_nodes = None
    # Data variables
    self.classes = None
    # Variables for the AL algorithm
    self.initialized = False
    x_train, y_train = train_data
    x_pool, y_pool = pool_data
    self.X = np.concatenate((x_train, x_pool), axis=0)
    self.y = np.concatenate((y_train, y_pool), axis=0)
    classes = list(set(self.y))
    self.n_classes = len(classes)
    self.already_clustered = False
    start_time = time.time()
    if self.max_features is not None and self.X.shape[1] > self.max_features:
      pca  = PCA(n_components=self.max_features)
      samples = np.copy(self.X)
      np.random.shuffle(samples)
      samples = samples[:int(0.1*self.X.shape[0])]
      pca.fit(samples)
      self.transformed_X = np.dot(self.X, pca.components_.T)

      #connectivity = kneighbors_graph(self.transformed_X,max_features)
      self.model = AgglomerativeClustering(
          affinity=self.affinity, linkage=self.linkage, n_clusters=len(classes))
      print("Compute agglomerative clustering...")
      self.fit_cluster(self.transformed_X)
    else:
      self.model = AgglomerativeClustering(
          affinity=self.affinity, linkage=self.linkage, n_clusters=len(classes))
      print("Compute agglomerative clustering...")

      self.fit_cluster(self.X)

    t = time.time() - start_time
    print('Clustering done in {}m...'.format(t/60))
    self.y_labels = {}
    # Fit cluster and update cluster variables

    self.create_tree()
    print('Finished creating hierarchical cluster')

  def fit_cluster(self, X):
    if not self.already_clustered:
      self.model.fit(X)
      self.already_clustered = True
    self.n_leaves = self.model.n_leaves_
    # self.n_components = self.model.n_components_
    self.children_list = self.model.children_

  def create_tree(self):
    node_dict = {}
    for i in range(self.n_leaves):
      node_dict[i] = [None, None]
    for i in range(len(self.children_list)):
      node_dict[self.n_leaves + i] = self.children_list[i]
    self.node_dict = node_dict
    # The sklearn hierarchical clustering algo numbers leaves which correspond
    # to actual datapoints 0 to n_points - 1 and all internal nodes have
    # ids greater than n_points - 1 with the root having the highest node id
    self.root = max(self.node_dict.keys())
    self.tree = Tree(self.root, self.node_dict)
    self.tree.create_child_leaves_mapping(range(self.n_leaves))
    for v in node_dict:
      self.admissible[v] = set()

  def get_child_leaves(self, node):
    return self.tree.get_child_leaves(node)

  def get_node_leaf_counts(self, node_list):
    node_counts = []
    for v in node_list:
      node_counts.append(len(self.get_child_leaves(v)))
    return np.array(node_counts)

  def get_class_counts(self, y):
    """Gets the count of all classes in a sample.
    Args:
      y: sample vector for which to perform the count
    Returns:
      count of classes for the sample vector y, the class order for count will
      be the same as that of self.classes
    """
    unique, counts = np.unique(y, return_counts=True)
    complete_counts = []
    for c in self.classes:
      if c not in unique:
        complete_counts.append(0)
      else:
        index = np.where(unique == c)[0][0]
        complete_counts.append(counts[index])
    return np.array(complete_counts)

  def observe_labels(self, labeled):
    for i in labeled:
      self.y_labels[i] = labeled[i]
    self.classes = np.array(
        sorted(list(set([self.y_labels[k] for k in self.y_labels]))))
    self.n_classes = len(self.classes)

  def initialize_algo(self):
    self.pruning = [self.root]
    self.labels[self.root] = np.random.choice(self.classes)
    node = self.tree.get_node(self.root)
    node.best_label = self.labels[self.root]
    self.selected_nodes = [self.root]

  def get_node_class_probabilities(self, node, y=None):
    children = self.get_child_leaves(node)
    if y is None:
      y_dict = self.y_labels
    else:
      y_dict = dict(zip(range(len(y)), y))
    labels = [y_dict[c] for c in children if c in y_dict]
    # If no labels have been observed, simply return uniform distribution
    if len(labels) == 0:
      return 0, np.ones(self.n_classes)/self.n_classes
    return len(labels), self.get_class_counts(labels) / (len(labels) * 1.0)

  def get_node_upper_lower_bounds(self, node):
    n_v, p_v = self.get_node_class_probabilities(node)
    # If no observations, return worst possible upper lower bounds
    if n_v == 0:
      return np.zeros(len(p_v)), np.ones(len(p_v))
    delta = 1. / n_v + np.sqrt(p_v * (1 - p_v) / (1. * n_v))
    return (np.maximum(p_v - delta, np.zeros(len(p_v))),
            np.minimum(p_v + delta, np.ones(len(p_v))))

  def get_node_admissibility(self, node):
    p_lb, p_up = self.get_node_upper_lower_bounds(node)
    all_other_min = np.vectorize(
        lambda i:min([1 - p_up[c] for c in range(len(self.classes)) if c != i]))
    lowest_alternative_error = self.beta * all_other_min(
        np.arange(len(self.classes)))
    return 1 - p_lb < lowest_alternative_error

  def get_adjusted_error(self, node):
    _, prob = self.get_node_class_probabilities(node)
    error = 1 - prob
    admissible = self.get_node_admissibility(node)
    not_admissible = np.where(admissible != True)[0]
    error[not_admissible] = 1.0
    return error

  def get_class_probability_pruning(self, method='lower'):
    prob_pruning = []
    for v in self.pruning:
      label = self.labels[v]
      label_ind = np.where(self.classes == label)[0][0]
      if method == 'empirical':
        _, v_prob = self.get_node_class_probabilities(v)
      else:
        lower, upper = self.get_node_upper_lower_bounds(v)
        if method == 'lower':
          v_prob = lower
        elif method == 'upper':
          v_prob = upper
        else:
          raise NotImplementedError
      prob = v_prob[label_ind]
      prob_pruning.append(prob)
    return np.array(prob_pruning)

  def get_pruning_impurity(self, y):
    impurity = []
    for v in self.pruning:
      _, prob = self.get_node_class_probabilities(v, y)
      impurity.append(1-max(prob))
    impurity = np.array(impurity)
    weights = self.get_node_leaf_counts(self.pruning)
    weights = weights / sum(weights)
    return sum(impurity*weights)

  def update_scores(self):
    node_list = set(range(self.n_leaves))
    # Loop through generations from bottom to top
    while len(node_list) > 0:
      parents = set()
      for v in node_list:
        node = self.tree.get_node(v)
        # Update admissible labels for node
        admissible = self.get_node_admissibility(v)
        admissable_indices = np.where(admissible)[0]
        for l in self.classes[admissable_indices]:
          self.admissible[v].add(l)
        # Calculate score
        v_error = self.get_adjusted_error(v)
        best_label_ind = np.argmin(v_error)
        if admissible[best_label_ind]:
          node.best_label = self.classes[best_label_ind]
        score = v_error[best_label_ind]
        node.split = False

        # Determine if node should be split
        if v >= self.n_leaves:  # v is not a leaf
          if len(admissable_indices) > 0:  # There exists an admissible label
            # Make sure label set for node so that we can flow to children
            # if necessary
            assert node.best_label is not None
            # Only split if all ancestors are admissible nodes
            # This is part  of definition of admissible pruning
            admissible_ancestors = [len(self.admissible[a]) > 0 for a in
                                    self.tree.get_ancestor(node)]
            if all(admissible_ancestors):
              left = self.node_dict[v][0]
              left_node = self.tree.get_node(left)
              right = self.node_dict[v][1]
              right_node = self.tree.get_node(right)
              node_counts = self.get_node_leaf_counts([v, left, right])
              split_score = (node_counts[1] / node_counts[0] *
                             left_node.score + node_counts[2] /
                             node_counts[0] * right_node.score)
              if split_score < score:
                score = split_score
                node.split = True
        node.score = score
        if node.parent:
          parents.add(node.parent.name)
        node_list = parents

  def update_pruning_labels(self):
    for v in self.selected_nodes:
      node = self.tree.get_node(v)
      pruning = self.tree.get_pruning(node)
      self.pruning.remove(v)
      self.pruning.extend(pruning)
    # Check that pruning covers all leave nodes
    node_counts = self.get_node_leaf_counts(self.pruning)
    assert sum(node_counts) == self.n_leaves
    # Fill in labels
    for v in self.pruning:
      node = self.tree.get_node(v)
      if node.best_label  is None:
        node.best_label = node.parent.best_label
      self.labels[v] = node.best_label

  def get_fake_labels(self):
    fake_y = np.zeros(self.X.shape[0])
    for p in self.pruning:
      indices = self.get_child_leaves(p)
      fake_y[indices] = self.labels[p]
    return fake_y

  def train_using_fake_labels(self, model, X_test, y_test):
    classes_labeled = set([self.labels[p] for p in self.pruning])
    if len(classes_labeled) == self.n_classes:
      fake_y = self.get_fake_labels()
      model.fit(self.X, fake_y)
      test_acc = model.score(X_test, y_test)
      return test_acc
    return 0

  def select_batch_(self, N, already_selected, labeled, y, **kwargs):
    # Observe labels for previously recommended batches
    self.observe_labels(labeled)

    if not self.initialized:
      self.initialize_algo()
      self.initialized = True
      print('Initialized algo')

    print('Updating scores and pruning for labels from last batch')
    self.update_scores()
    self.update_pruning_labels()
    print('Nodes in pruning: %d' % (len(self.pruning)))
    print('Actual impurity for pruning is: %.2f' %
          (self.get_pruning_impurity(y)))

    # TODO(lishal): implement multiple selection methods
    selected_nodes = set()
    weights = self.get_node_leaf_counts(self.pruning)
    probs = 1 - self.get_class_probability_pruning()
    weights = weights * probs
    weights = weights / sum(weights)
    batch = []

    print('Sampling batch')
    while len(batch) < N:
      node = np.random.choice(list(self.pruning), p=weights)
      children = self.get_child_leaves(node)
      children = [
          c for c in children if c not in self.y_labels and c not in batch
      ]
      if len(children) > 0:
        selected_nodes.add(node)
        batch.append(np.random.choice(children))
    self.selected_nodes = selected_nodes
    return batch

  def to_dict(self):
    output = {}
    output['node_dict'] = self.node_dict
    return output

  def __call__(self, model, pool, train_data):

    if self.subsample:
        x_pool, pool_labels = pool
        subsample_num = int(self.subsample*x_pool.shape[0])
        print('Subsampling... {} samples'.format(subsample_num))
        subsample_idx = np.random.choice(np.arange(x_pool.shape[0]), subsample_num, replace=False)
        x_pool = x_pool[subsample_idx]
        pool_labels = pool_labels[subsample_idx]
        pool = x_pool, pool_labels

    if self.init_at_each_step:
        self.initialize_hierarchy(train_data, pool)

    _, y_train = train_data
    _, y_pool = pool
    already_selected = np.arange(len(y_train))
    labeled = {}
    for selected in already_selected:
      labeled[selected] = y_train[selected]
    y = self.y
    indices = self.select_batch_(self.n_px, already_selected, labeled, y)
    indices = np.array(indices) - len(y_train)

    if self.subsample:
        indices = subsample_idx[indices]

    return indices

#===============================================================================
#                          Performance based Active Learning
#===============================================================================

class LAL(Query):
    def __init__(self, n_px, hyperparams, shuffle_prop):
        super().__init__(n_px, hyperparams,
            shuffle_prop=shuffle_prop, reverse=True
        )

    def compute_score(self, model, pool):
        data_loader = self.pool_loader(pool)
        pred_error_reduction = self.compute_probs(model, data_loader)
        self.score = pred_error_reduction
        return pred_error_reduction

#===============================================================================
#                                  Utils
#===============================================================================
def _shuffle_subset(data: np.ndarray, shuffle_prop: float) -> np.ndarray:
    to_shuffle = np.nonzero(np.random.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[np.random.permutation(to_shuffle), ...]
    return data

def get_indices_as_array(img_shape: tuple) -> np.ndarray:
    indices = np.zeros(img_shape)
    indices = np.where(indices == 0)
    indices = np.array(indices)
    return indices.T

def gather_expand(data, dim, index):
    """
    Gather indices `index` from `data` after expanding along dimension `dim`.

    Args:
        data (tensor): A tensor of data.
        dim (int): dimension to expand along.
        index (tensor): tensor with the indices to gather.

    References:
        Code from https://github.com/BlackHC/BatchBALD/blob/master/src/torch_utils.py

    Returns:
        Tensor with the same shape as `index`.
    """
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)

class Node(object):
  """Node class for hierarchical clustering.
  Initialized with name and left right children.
  """

  def __init__(self, name, left=None, right=None):
    self.name = name
    self.left = left
    self.right = right
    self.is_leaf = left is None and right is None
    self.parent = None
    # Fields for hierarchical clustering AL
    self.score = 1.0
    self.split = False
    self.best_label = None
    self.weight = None

  def set_parent(self, parent):
    self.parent = parent


class Tree(object):
  """Tree object for traversing a binary tree.
  Most methods apply to trees in general with the exception of get_pruning
  which is specific to the hierarchical clustering AL method.
  """

  def __init__(self, root, node_dict):
    """Initializes tree and creates all nodes in node_dict.
    Args:
      root: id of the root node
      node_dict: dictionary with node_id as keys and entries indicating
        left and right child of node respectively.
    """
    self.node_dict = node_dict
    self.root = self.make_tree(root)
    self.nodes = {}
    self.leaves_mapping = {}
    self.fill_parents()
    self.n_leaves = None

  def print_tree(self, node, max_depth):
    """Helper function to print out tree for debugging."""
    node_list = [node]
    output = ""
    level = 0
    while level < max_depth and len(node_list):
      children = set()
      for n in node_list:
        node = self.get_node(n)
        output += ("\t"*level+"node %d: score %.2f, weight %.2f" %
                   (node.name, node.score, node.weight)+"\n")
        if node.left:
          children.add(node.left.name)
        if node.right:
          children.add(node.right.name)
      level += 1
      node_list = children
    return print(output)

  def make_tree(self, node_id):
    if node_id is not None:
      return Node(node_id,
                  self.make_tree(self.node_dict[node_id][0]),
                  self.make_tree(self.node_dict[node_id][1]))

  def fill_parents(self):
    # Setting parent and storing nodes in dict for fast access
    def rec(pointer, parent):
      if pointer is not None:
        self.nodes[pointer.name] = pointer
        pointer.set_parent(parent)
        rec(pointer.left, pointer)
        rec(pointer.right, pointer)
    rec(self.root, None)

  def get_node(self, node_id):
    return self.nodes[node_id]

  def get_ancestor(self, node):
    ancestors = []
    if isinstance(node, int):
      node = self.get_node(node)
    while node.name != self.root.name:
      node = node.parent
      ancestors.append(node.name)
    return ancestors

  def fill_weights(self):
    for v in self.node_dict:
      node = self.get_node(v)
      node.weight = len(self.leaves_mapping[v]) / (1.0 * self.n_leaves)

  def create_child_leaves_mapping(self, leaves):
    """DP for creating child leaves mapping.

    Storing in dict to save recompute.
    """
    self.n_leaves = len(leaves)
    for v in leaves:
      self.leaves_mapping[v] = [v]
    node_list = set([self.get_node(v).parent for v in leaves])
    while node_list:
      to_fill = copy.copy(node_list)
      for v in node_list:
        if (v.left.name in self.leaves_mapping
            and v.right.name in self.leaves_mapping):
          to_fill.remove(v)
          self.leaves_mapping[v.name] = (self.leaves_mapping[v.left.name] +
                                         self.leaves_mapping[v.right.name])
          if v.parent is not None:
            to_fill.add(v.parent)
      node_list = to_fill
    self.fill_weights()

  def get_child_leaves(self, node):
    return self.leaves_mapping[node]

  def get_pruning(self, node):
    if node.split:
      return self.get_pruning(node.left) + self.get_pruning(node.right)
    else:
      return [node.name]

class QueryOutput:
    """
    Class that restore the results of the query
    """
    def __init__(self, dataset, history, classes, config):
        self.dataset = dataset
        self.history = history
        self.classes  = classes
        self.config = config

        coordinates = np.array(history['coordinates'][:(config['step']-1)*config['n_px']])
        if len(coordinates) > 0:
            coordinates = tuple((coordinates[:,0], coordinates[:,1]))
            self.dataset.train_gt.gt[coordinates] = np.array(history['labels'][:(config['step']-1)*config['n_px']])
        self.dataset.label_values = [item['label'] for item in self.classes.values()]
        self.dataset.n_classes = len(self.dataset.label_values)
        self.coordinates = np.array(history['coordinates'][-config['n_px']:])
        self.step_ = config['step']
        self.res_dir = config['res_dir']
        self.timestamp = config['timestamp']

        self.n_added = 0
        self.n_px = config['n_px']

        self.annotation = None


    def patches_(self):
        self.regions = np.zeros_like(self.dataset.GT['train'])
        self.patch_id = 0
        self.patches, self.patch_coordinates = window(self.dataset.IMG, self.coordinates)
        self.regions, _ = window(self.regions, self.coordinates)

    def update_classes(self, new_label_id):
        self.classes[new_label_id]['added_px'] += 1

    def label(self):
        self.history['labels'].append(self.annotation)
        self.dataset.train_gt.gt[self.x, self.y] = self.annotation
        self.update_classes(self.annotation)

    def add_class(self, new_class):
        self.dataset.label_values.append(new_class)
        self.dataset.n_classes += 1
        class_id = len(self.dataset.label_values) - 1
        self.classes[class_id] = {}
        self.classes[class_id]['label'] = new_class
        self.classes[class_id]['nb_px'] = 0
        self.classes[class_id]['added_px'] = 0
        self.classes[class_id]['pseudo_labels'] = 0

    def save(self):
        self.n_added += 1
        pkl.dump((self.dataset.train_gt, self.classes, self.history, self.config),\
          open(os.path.join(self.res_dir, 'oracle_{}_step_{}.pkl'.format(self.timestamp, self.step_)), 'wb'))


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

    if config['query'] == 'vaal':
        config.setdefault('num_adv_steps', 1)
        config.setdefault('num_vae_steps', 2)
        config.setdefault('adversary_param', 1)
        config.setdefault('beta', 1)
        config.setdefault('batch_size', 128)
        vae = VAE(dataset.n_bands, z_dim=8)
        discriminator = Discriminator(z_dim=8)
        model = VaalClassifier(vae, discriminator)
        query = AdversarialSampler(config['n_px'], config, shuffle_prop=0)

    elif config['query'] == 'batch_bald':
        from learning.consistent_dropout import MCConsistentDropoutModule
        config.setdefault('batch_size', 128)
        lr = config.setdefault('learning_rate', 0.01)
        config.setdefault('num_samples', 30)
        config.setdefault('num_draw', 100)
        net = HuEtAl(dataset.n_bands, dataset.n_classes, dropout=0.5)
        net = MCConsistentDropoutModule(net)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=config['weight_decay'])
        config['optimizer'] = optimizer
        criterion = nn.CrossEntropyLoss(weight=config['weights'])
        config.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=config['epochs']//4, verbose=True))
        model = BayesianModelWrapper(net, criterion)
        query = BatchBALD(config['n_px'], config, shuffle_prop=0)

    elif config['query'] == 'bald':
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

    elif config['query'] == 'random':
        model = Classifier()
        query = RandomSampling(config['n_px'], config, shuffle_prop=0)

    elif config['query'] == 'multi_view':
        config.setdefault('batch_size', 128)
        config.setdefault('weight_decay', 0)
        lr = config.setdefault('learning_rate', 0.01)
        net = HuEtAl(dataset.n_bands, dataset.n_classes)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss(weight=config['weights'])
        model = MultiView(net, optimizer, criterion)
        query = BatchBALD(config['n_px'], config, shuffle_prop=0)

    elif config['query'] == 'hierarchical':
        model = Classifier()
        config.setdefault('beta', 2)
        query = HierarchicalClusterAL(config['n_px'], config, 0, dataset, 3)

    elif config['query'] == 'lal':
        config.setdefault('Q', 10)
        config.setdefault('tau', [0.15, 0.1, 0.2, 0.3])
        config.setdefault('M', 10)
        config.setdefault('batch_size', 128)
        model = LalRegressor(nEstimators=50)
        query = LAL(config['n_px'], config, 0)

    elif config['query'] == 'variance':
        config.setdefault('batch_size', 128)
        config.setdefault('learning_rate', 0.001)
        config.setdefault('num_samples', 10)
        model = BayesianModel(dataset.n_bands, dataset.n_classes)
        query = Variance(config['n_px'], config, shuffle_prop=0)


    return model, query, config
