
import io
import numpy as np
import sys
import tensorflow as tf

import matplotlib
backend = 'Agg' if sys.platform == 'linux' else 'TkAgg'
matplotlib.use(backend)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _build_network(
        name,
        inputs, 
        hidden_layer_dims,
        output_dim,
        activation_fn=tf.nn.relu,
        bias_initializer=tf.constant_initializer(0.1),
        output_nonlinearity=None,
        dropout_keep_prob=1.,
        weights_regularizer=None,
        reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        hidden = inputs
        for hidden_dim in hidden_layer_dims:
            hidden = tf.contrib.layers.fully_connected(
                        hidden, 
                        hidden_dim, 
                        activation_fn=activation_fn,
                        biases_initializer=bias_initializer,
                        weights_regularizer=weights_regularizer)
            hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        output = tf.contrib.layers.fully_connected(
            hidden, 
            output_dim, 
            activation_fn=output_nonlinearity,
            biases_initializer=tf.constant_initializer(0.)
        )
        return output

def _build_train_op(loss, learning_rate, var_list, grad_scale, scope, global_step=None):
    with tf.variable_scope(scope):
        opt = tf.train.AdamOptimizer(learning_rate)
        grads = tf.gradients(loss, var_list)
        scaled_grads, _ = tf.clip_by_global_norm(grads, grad_scale)
        train_op = opt.apply_gradients([(g,v) for (g,v) in zip(scaled_grads, var_list)], 
            global_step=global_step)

    # summaries
    summaries = []
    summaries += [tf.summary.scalar('{}/global_grad_norm'.format(scope), tf.global_norm(grads))]
    summaries += [tf.summary.scalar('{}/global_scaled_grad_norm'.format(scope), tf.global_norm(scaled_grads))]
    summaries += [tf.summary.scalar('{}/global_var_norm'.format(scope), tf.global_norm(var_list))]
    
    return train_op, summaries

def plot_grid_summary(xs, name='', side=28):
    nrow, ncol = np.shape(xs)[:2]
    
    fig = plt.figure(figsize=(ncol+1, nrow+1)) 

    gs = gridspec.GridSpec(nrow, ncol,
        wspace=0.0, hspace=0.0, 
        top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
        left=0.5/(ncol+1), right=1-0.5/(ncol+1)
    ) 
    
    for i, x in enumerate(xs):
        for j, img in enumerate(x):
            ax = plt.subplot(gs[i,j])
            ax.imshow(img.reshape(side,side))
            ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_sum = tf.Summary.Image(
        encoded_image_string=buf.getvalue(), 
        height=side * nrow, 
        width=side * ncol
    )

    summary = tf.Summary(value=[
       tf.Summary.Value(tag='gen/{}'.format(name), image=img_sum)
    ])
    return summary