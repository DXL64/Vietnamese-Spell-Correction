from torch.utils.data.dataloader import Sampler
import sys
sys.path.append("..")
from data.spellcorrect_dataset import SpellCorrectDataset
import numpy as np
import copy
from tqdm import tqdm
import time

class RandomBatchSampler(Sampler):
    def __init__(self, data: SpellCorrectDataset, batch_size = 1, shuffle = True, RANDOM_SEED = 1301, MAXIMUM_TOKENS_PER_BATCH = 1024):
        self.data = data
        self.seq = list(range(0, len(self.data)))
        self.shuffle = shuffle
        self.iters = 0
        self.batch_size = batch_size
        self.RANDOM_SEED = RANDOM_SEED
        self.MAXIMUM_TOKENS_PER_BATCH = MAXIMUM_TOKENS_PER_BATCH
        if self.shuffle:
            np.random.seed(self.RANDOM_SEED)
            np.random.shuffle(self.seq)
        self.seq = [ self.seq[index: index + self.batch_size] \
            for index in range(self.iters, len(self.seq), self.batch_size)]
        self.default_seq = copy.deepcopy(self.seq)

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)
    
    def load_checkpoints(self, iters = 0):
        self.seq = list(range(0, len(self.data)))
        if self.shuffle: 
            np.random.seed(self.RANDOM_SEED)
            np.random.shuffle(self.seq)
        self.iters = iters
        self.seq = [ self.seq[index: index + self.batch_size] \
            for index in range(self.iters, len(self.seq), self.batch_size)]

class BucketBatchSampler(Sampler):
    def __init__(self, data: SpellCorrectDataset, shuffle = True, RANDOM_SEED = 1301, MAXIMUM_TOKENS_PER_BATCH = 1024):
        start = time.time()
        self.remained_indies = None
        self.data = data
        self.shuffle = shuffle
        self.RANDOM_SEED = RANDOM_SEED
        self.MAXIMUM_TOKENS_PER_BATCH = MAXIMUM_TOKENS_PER_BATCH
        print("Initializing Bucket Batch Sampler From Scratch")
        # self.data.dataset = sorted(self.data.dataset, key = lambda x: x[2])
        token_counts = 0
        indies_lists = []
        self.seq = []
        for index, values in tqdm(enumerate(self.data.dataset)):
            if token_counts >= self.MAXIMUM_TOKENS_PER_BATCH:
                self.seq.append(indies_lists)
                indies_lists = []
                token_counts = 0
            indies_lists.append(index)
            token_counts += values[2]
        if len(indies_lists) != 0 and token_counts != 0:
            self.seq.append(indies_lists)

        if shuffle:
            np.random.seed(self.RANDOM_SEED)
            np.random.shuffle(self.seq)
        end = time.time()
        print(f"Initialized Bucket Batch Sampler From Scratch: {end - start}")
        self.default_seq = copy.deepcopy(self.seq)
        
    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def load_checkpoints(self, remained_indies):
        start = time.time()
        print("Loading Bucket Batch Sampler From Checkpoint")
        remained_indies = sorted(remained_indies)
        token_counts = 0
        indies_lists = []
        self.seq = []
        for index in tqdm(remained_indies):
            values = self.data.dataset[index]
            if token_counts >= self.MAXIMUM_TOKENS_PER_BATCH:
                self.seq.append(indies_lists)
                indies_lists = []
                token_counts = 0
            indies_lists.append(index)
            token_counts += values[2]

        if len(indies_lists) != 0 and token_counts != 0:
            self.seq.append(indies_lists)
        
        if self.shuffle:
            np.random.seed(self.RANDOM_SEED)
            np.random.shuffle(self.seq)
        end = time.time()
        print(f"Loaded Bucket Batch Sampler From Checkpoint: {end - start}")


