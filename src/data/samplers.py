import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import copy
import torch
import random
import numpy as np

from collections import defaultdict


class RandomIdentitySampler(torch.utils.data.Sampler):
    def __init__(self, datasource, batch_size=1, num_instances=1, index_dict=None):
        self.datasource = datasource

        if batch_size <= num_instances:
            raise ValueError("batch_size <= num_instances")

        if batch_size % num_instances != 0:
            raise ValueError("batch_size % num_instances != 0")

        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        if index_dict == None:
            self.index_dict = defaultdict(list)
            for index, (_, person_id, _) in enumerate(self.datasource):
                self.index_dict[person_id].append(index)
        else:
            self.index_dict = index_dict

        self.person_ids = list(self.index_dict.keys())

        self.length = 0
        for person_id in self.person_ids:
            num = len(self.index_dict[person_id])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __len__(self):
        return self.length

    def __iter__(self):
        batch_idx_dict = defaultdict(list)

        for person_id in self.person_ids:
            idxs = copy.deepcopy(self.index_dict[person_id])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            random.shuffle(idxs)

            batch_idxs = []

            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idx_dict[person_id].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.person_ids)
        batch = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for person_id in selected_pids:
                batch.extend(batch_idx_dict[person_id].pop(0))
                if len(batch_idx_dict[person_id]) == 0:
                    avai_pids.remove(person_id)

        self.length = len(batch)

        return iter(batch)
