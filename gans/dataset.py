
import numpy as np

from utils import compute_n_batches, compute_batch_idxs

class Dataset(object):
    
    def __init__(self, x, batch_size, shuffle=True):
        self.x = x
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(x)
        self.n_batches = compute_n_batches(self.n_samples, self.batch_size)
    
    def _shuffle(self):
        if self.shuffle:
            idxs = np.random.permutation(self.n_samples)
            self.x = self.x[idxs]
    
    def batches(self):
        self._shuffle()
        for bidx in range(self.n_batches):
            idxs = compute_batch_idxs(bidx * self.batch_size, self.batch_size, self.n_samples)
            yield dict(x=self.x[idxs])