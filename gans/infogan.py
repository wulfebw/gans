
import collections
import numpy as np
import sys
import tensorflow as tf

from tf_utils import _build_network, _build_train_op, plot_grid_summary
from utils import compute_n_batches, compute_batch_idxs

class InfoGAN(object):
    def __init__(
            self,
            input_dim=28 ** 2,
            z_dim=64,
            c_dim=10,
            dropout_keep_prob=1.,
            gen_hidden_layer_dims=(64,128,256,512),
            critic_hidden_layer_dims=(512,256,128,64),
            recog_hidden_layer_dims=(512,256,128,64),
            gp_weight=10,
            recog_weight=1,
            gen_learning_rate=5e-4,
            critic_learning_rate=2.5e-4,
            recog_learning_rate=5e-4,
            grad_scale=50):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dropout_keep_prob = dropout_keep_prob
        self.gen_hidden_layer_dims = gen_hidden_layer_dims
        self.critic_hidden_layer_dims = critic_hidden_layer_dims
        self.recog_hidden_layer_dims = recog_hidden_layer_dims
        self.gp_weight = gp_weight
        self.recog_weight = recog_weight
        self.gen_learning_rate = gen_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.recog_learning_rate = recog_learning_rate
        self.grad_scale = grad_scale
        self._build_model()
    
    def _build_model(self):
        self._build_placeholders()
        self._build_generator()
        self._build_critic()
        self._build_recognition()
        self._build_loss()
        self._build_train_op()
        self._build_summary_op()
    
    def _build_placeholders(self):
        self.rx = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='rx')
        self.z = tf.placeholder(tf.float32, shape=(None, self.z_dim), name='z')
        self.c = tf.placeholder(tf.float32, shape=(None, self.c_dim), name='c')
        self.eps = tf.placeholder(tf.float32, shape=(None, 1), name='eps')
        self.dropout_keep_prob_ph = tf.placeholder_with_default(
            self.dropout_keep_prob, 
            shape=(), 
            name='dropout_keep_prob_ph'
        )
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
    def _build_generator(self):
        hidden = tf.concat((self.z, self.c), axis=1)
        self.gx = _build_network(
            'gen',
            hidden,
            self.gen_hidden_layer_dims,
            self.input_dim,
            output_nonlinearity=tf.nn.sigmoid,
            dropout_keep_prob=self.dropout_keep_prob_ph
        )
        self.gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gen')

        # summaries
        self.gen_summaries = []
        side = int(np.sqrt(self.input_dim))
        max_outputs = 3
        imgs = tf.reshape(self.gx[:max_outputs], (max_outputs,side,side,1))
        self.gen_summaries += [tf.summary.image('gen/gx', imgs, max_outputs=max_outputs)]
        
    def _build_critic(self):
        rx, gx = self.rx, self.gx
        
        # gradient penalty
        xhat = self.eps * rx + (1 - self.eps) * gx
        xhat_outputs = _build_network(
            'critic',
            xhat,
            self.critic_hidden_layer_dims,
            1,
            dropout_keep_prob=self.dropout_keep_prob_ph
        )
        xhat_gradients = tf.gradients(xhat_outputs, xhat)[0]
        slopes = tf.sqrt(tf.reduce_sum(xhat_gradients ** 2, reduction_indices=[1]))
        self.gradient_penalty = self.gp_weight * tf.reduce_mean((slopes - 1) ** 2)
        
        # real prediction
        self.real_scores = _build_network(
            'critic',
            self.rx,
            self.critic_hidden_layer_dims,
            1,
            dropout_keep_prob=self.dropout_keep_prob_ph,
            reuse=True
        )
        
        # gen prediction
        self.gen_scores = _build_network(
            'critic',
            self.gx,
            self.critic_hidden_layer_dims,
            1,
            dropout_keep_prob=self.dropout_keep_prob_ph,
            reuse=True
        )
        
        self.critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'critic')

        # summaries 
        self.critic_summaries = []
        self.critic_summaries += [tf.summary.scalar('critic/gp', self.gradient_penalty)]
        
    def _build_recognition(self):
        self.recog_scores = _build_network(
            'recog',
            self.gx,
            self.recog_hidden_layer_dims,
            self.c_dim,
            dropout_keep_prob=self.dropout_keep_prob_ph
        )
        self.recog_probs = tf.nn.softmax(self.recog_scores)
        self.recog_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'recog')
        
    def _build_loss(self):
        # critic loss
        critic_real_loss = -tf.reduce_mean(self.real_scores)
        critic_gen_loss = tf.reduce_mean(self.gen_scores)
        self.critic_loss = critic_real_loss + critic_gen_loss + self.gradient_penalty
        
        # recog loss
        c_labels = tf.cast(tf.argmax(self.c, axis=1), tf.int32)
        self.recog_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=c_labels, logits=self.recog_scores))
        
        # gen loss
        self.gen_loss = -tf.reduce_mean(self.gen_scores)
        self.gen_loss += self.recog_weight * tf.reduce_mean(self.c * self.recog_probs)

        # summaries
        self.critic_summaries += [tf.summary.scalar('critic/w_dist', 
            -(critic_real_loss + critic_gen_loss))]
        self.gen_summaries += [tf.summary.scalar('gen/loss', self.gen_loss)]
        self.gen_summaries += [tf.summary.scalar('recog/loss', self.recog_loss)]

    def _build_train_op(self):
        self.critic_train_op, summaries = _build_train_op(
            self.critic_loss, 
            self.critic_learning_rate, 
            self.critic_vars, 
            self.grad_scale,
            'critic'
        )
        self.critic_summaries += summaries
        self.recog_train_op, summaries = _build_train_op(
            self.recog_loss, 
            self.recog_learning_rate, 
            self.recog_vars, 
            self.grad_scale,
            'recog'
        )
        self.gen_summaries += summaries
        self.gen_train_op, summaries = _build_train_op(
            self.gen_loss, 
            self.gen_learning_rate, 
            self.gen_vars, 
            self.grad_scale, 
            'gen',
            global_step=self.global_step
        )
        self.gen_summaries += summaries

    def _build_summary_op(self):
        self.critic_summary_op = tf.summary.merge(self.critic_summaries)
        self.gen_summary_op = tf.summary.merge(self.gen_summaries)
        
    def _sample_noise(self, n_samples):
        z = np.random.randn(n_samples, self.z_dim)
        c_idxs = np.random.randint(low=0, high=self.c_dim, size=n_samples)
        eye = np.eye(self.c_dim)
        c = eye[c_idxs]
        eps = np.random.uniform(0, 1, n_samples).reshape(-1, 1)
        return z, c, eps

    def _train_batch(
            self, 
            batch, 
            info, 
            writer, 
            bidx, 
            summary_every,
            train_gen_every):
        # common
        sess = tf.get_default_session()
        batch_size = batch['x'].shape[0]
        summarize = bidx % summary_every == 0
        
        # critic
        outputs = [
            self.critic_loss, 
            self.critic_train_op
        ]
        if summarize:
            outputs += [
                self.critic_summary_op,
                self.global_step
            ]
        z, c, eps = self._sample_noise(batch_size)
        feed = {
            self.rx: batch['x'],
            self.z: z,
            self.c: c,
            self.eps: eps
        }
        fetched = sess.run(outputs, feed_dict=feed)
        if summarize:
            critic_loss, _, summary, step = fetched
            if writer is not None:
                writer.add_summary(tf.Summary.FromString(summary), step)
                writer.flush()
        else:
            critic_loss, _ = fetched

        # update critic info
        info['critic_loss'] += critic_loss
        info['critic_itr'] += 1
        
        # gen
        if bidx % train_gen_every == 0:

            outputs = [
                self.gen_loss, 
                self.gen_train_op, 
                self.recog_loss, 
                self.recog_train_op
            ]
            if summarize:
                outputs += [self.gen_summary_op]
            z, c, _ = self._sample_noise(batch_size)
            feed = {
                self.z: z,
                self.c: c
            }
            fetched = sess.run(outputs, feed_dict=feed)
            if summarize:
                gen_loss, _, recog_loss, _, summary = fetched
                if writer is not None:
                    writer.add_summary(tf.Summary.FromString(summary), step)
                    writer.flush()
            else:
                gen_loss, _, recog_loss, _ = fetched
            
            # update gen info
            info['gen_loss'] += gen_loss
            info['recog_loss'] += recog_loss
            info['gen_itr'] += 1
        
    def _report(self, info, name, bidx, n_batches, epoch, n_epochs):
        msg = '\r{} epoch: {} / {} batch: {} / {}'.format(name, epoch+1, n_epochs, bidx+1, n_batches)
        msg += ' critic_loss: {:.5f} '.format(info['critic_loss'] / info['critic_itr'])
        msg += ' gen_loss: {:.5f} '.format(info['gen_loss'] / info['gen_itr'])
        msg += ' recog_loss: {:.5f} '.format(info['recog_loss'] / info['gen_itr'])
        sys.stdout.write(msg)

    def _validate(self, writer, step, permute=None, n_z=8):
        xs = []
        for i in range(n_z):
            c = np.eye(10)[np.arange(10)]
            z = np.tile(np.random.rand(self.z_dim), (10,1))
            info = self.generate(z=z, c=c)
            xs.append(info['gx'])

        summary = plot_grid_summary(xs, name='validation')
        writer.add_summary(summary, step)
        writer.flush()

        if permute is not None:
            xs_unpermute = []
            for x in xs:
                x_unpermute = np.zeros(x.shape)
                x_unpermute[:,permute] = x
                xs_unpermute.append(x_unpermute)
            summary = plot_grid_summary(xs_unpermute, name='validation_unpermute')
            writer.add_summary(summary, step)
            writer.flush()

    def train(
            self, 
            dataset, 
            val_dataset=None,
            n_epochs=10,
            train_gen_every=5,
            writer=None,
            summary_every=10,
            permute=None):
        for epoch in range(n_epochs):
            info = collections.defaultdict(float)
            for bidx, batch in enumerate(dataset.batches()):
                self._train_batch(batch, info, writer, bidx, summary_every, train_gen_every)
                self._report(info, 'train', bidx, dataset.n_batches, epoch, n_epochs)
            self._validate(writer, epoch, permute)
                
    def generate(self, z=None, c=None, batch_size=100):
        # setup 
        sess = tf.get_default_session()
        
        # sample noise if not provided
        if z is None and c is None:
            z, c, _ = self._sample_noise(batch_size)
        elif z is None:
            batch_size = len(c)
            z, _, _ = self._sample_noise(batch_size)
        else:
            batch_size = len(c)
        
        # at this point, z dictates number of samples
        n_samples = len(z)
        n_batches = compute_n_batches(n_samples, batch_size)
        
        # allocate return containers
        gx = np.zeros((n_samples, self.input_dim))
        scores = np.zeros((n_samples,1))

        # formulate outputs
        outputs = [self.gx, self.gen_scores]

        # run the batches
        for bidx in range(n_batches):
            idxs = compute_batch_idxs(bidx * batch_size, batch_size, n_samples)
            feed = {
                self.z: z[idxs],
                self.c: c[idxs]
            }
            fetched = sess.run(outputs, feed_dict=feed)

            # unpack
            gx[idxs] = fetched[0]
            scores[idxs] = fetched[1]

        # return the relevant info
        return dict(gx=gx, scores=scores)
