
import numpy as np
from sklearn.datasets import fetch_mldata

def compute_n_batches(n_samples, batch_size):
    n_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        n_batches += 1
    return n_batches

def compute_batch_idxs(start, batch_size, size, fill='random'):
    if start >= size:
        return list(np.random.randint(low=0, high=size, size=batch_size))
    
    end = start + batch_size

    if end <= size:
        return list(range(start, end))

    else:
        base_idxs = list(range(start, size))
        if fill == 'none':
            return base_idxs
        elif fill == 'random':
            remainder = end - size
            idxs = list(np.random.randint(low=0, high=size, size=remainder))
            return base_idxs + idxs
        else:
            raise ValueError('invalid fill: {}'.format(fill))

def load_data(side=28):
    mnist = fetch_mldata('MNIST original')
    x = mnist.data.astype('float64')
    y = mnist.target
    # permute order of samples
    data_permutation = np.random.permutation(len(x))
    x = x[data_permutation]
    y = y[data_permutation]
    # normalize x
    x /= 255.
    # also build a random permutation version
    img_permutation = np.random.permutation(side ** 2)
    x_permute = np.copy(x)[:,img_permutation]
    x_unpermute = np.zeros(x.shape)
    x_unpermute[:,img_permutation] = x_permute

    return dict(
        x=x,
        y=y,
        img_permutation=img_permutation,
        x_permute=x_permute,
        x_unpermute=x_unpermute
    )