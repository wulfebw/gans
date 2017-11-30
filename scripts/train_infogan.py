
import argparse
import numpy as np
import sys
import tensorflow as tf

sys.path.append('../gans')

import dataset
import infogan
import utils

side = 28

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='normal', required=False)
    args = parser.parse_args()

    data = utils.load_data(side=side)

    if args.mode == 'normal':
        x = data['x']
        permute = None
        summary_dir = '../data/summaries/normal'
    elif args.mode == 'permute':
        x = data['x_permute']
        permute = data['img_permutation']
        summary_dir = '../data/summaries/permute'

    writer = tf.summary.FileWriter(summary_dir)

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = infogan.InfoGAN(
            dropout_keep_prob=.95,
            recog_weight=1
        )
        sess.run(tf.global_variables_initializer())
        dataset = dataset.Dataset(x, batch_size=1000)
        model.train(dataset, n_epochs=1000, writer=writer, permute=permute)
