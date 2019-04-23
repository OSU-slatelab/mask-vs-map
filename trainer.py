"""
Trainer creates train ops and goes through all data to train or test.

Author: Peter Plantinga
Date: Fall 2017
"""

import tensorflow as tf
import time
import sys
from random import random
import numpy as np

def magnitude(complex_spec):
    return tf.sqrt(complex_spec[:,0] ** 2 + complex_spec[:,1] ** 2)

class Trainer:
    """ Train a model """

    def __init__(self, models,
        learn_rate  = 1e-4,
        lr_decay    = 1.,
        max_norm    = 5.0,
        loss_weight = {'fidelity': 1.0},
        lengths     = False,
        verbose     = False,
        batch_size  = 1,
    ):
        """ 
        Parameters
        ----------
        models : dict
            All models need for training in the format:
            {<name of model>: {'model': model, 'train': bool, 'vars': list}}
        learn_rate : float
            Rate of gradient descent
        lr_decay : float
            Amount of decay for learning rate
        max_norm : float
            For clipping norm
        loss_weight : dict
            How much weight to assign to each loss, of the form:
            {<name of loss>: float, ...}
        lengths : bool
            Whether to pass the length of the input to the fd, for recurrent models.
        verbose : bool
            Whether to print debugging info
        """
        
        self.verbose = verbose
        self.lengths = lengths
        self.learn_rate = learn_rate
        self.batch_size = batch_size

        if verbose:
            print("Generating compute graph")

        self.feed_dict = {}

        # Collect variables for training
        var_list = []
        self.training = []
        for model in models:
            if models[model]['train']:
                var_list += models[model]['vars']
                self.training.append(models[model]['model'].training)
            else:
                self.feed_dict[models[model]['model'].training] = False

        self.initialize_inputs(models, loss_weight)

        # Losses
        self.loss = 0
        self.losses = {}

        for loss in loss_weight:
            op = self.get_op(loss, models)
            self.losses[loss] = {'op': op }
            self.loss += loss_weight[loss] * op

        # Create primary train op
        self.losses['average'] = {'op': self.loss}
        self.train_op =  self._create_train_op(self.loss, var_list, max_norm, self.learn_rate, lr_decay)

    def initialize_inputs(self, models, loss_weight):
        """ Create placeholders for all types of inputs """
        self.irm, self.ibm_x, self.ibm_n, self.senone, self.trans = None, None, None, None, None
        if 'generator' in models:

            self.noisy = models['generator']['model'].inputs
            if 'masking' in loss_weight:
                shape = models['generator']['model'].outputs.get_shape().as_list()
                self.irm = tf.placeholder(tf.float32, shape = shape, name = 'irm')
                self.clean = None

            if 'speech-mask' in loss_weight:
                self.ibm_x = tf.placeholder(tf.float32, shape = shape, name = 'ibm_x')

            if 'noise-mask' in loss_weight:
                self.ibm_n = tf.placeholder(tf.float32, shape = shape, name = 'ibm_n')


            if 'teacher' in models:
                self.clean = models['teacher']['model'].inputs
            elif 'fidelity' in loss_weight:
                shape = models['generator']['model'].outputs.get_shape().as_list()
                self.clean = tf.placeholder(tf.float32, shape = shape, name = "clean")
            else:
                self.clean = None

            self.senone = None
        else:
            self.clean = models['student']['model'].inputs
            self.noisy = None

        if 'ctc' in loss_weight:
            self.trans = tf.sparse_placeholder(tf.int32)

        # The only time we need a senone placeholder is if theres a cross-entropy loss
        if 'cross-ent' in loss_weight:
            shape = models['student']['model'].outputs.get_shape().as_list()[:-1]
            self.senone = tf.placeholder(tf.int32, shape, name = "senone")


    def get_op(self, loss, models):
        """ This function creates many types of losses,
        
        Parameters
        ----------
        * loss : string
            The type of loss to create
        * models : dict
            The models to use for making the loss
        """

        if loss == 'cross-ent':
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = self.senone,
                logits = models['student']['model'].outputs,
            ))

        elif loss == 'ctc':
            return tf.reduce_mean(tf.nn.ctc_loss(
                labels = self.trans,
                inputs = models['student']['model'].outputs,
                sequence_length = [models['student']['model'].seq_len],
            ))

        elif loss == 'cer':
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(
                inputs = models['student']['model'].outputs, 
                sequence_length = [models['student']['model'].seq_len],
            )
            return tf.reduce_mean(tf.edit_distance(
                hypothesis = tf.cast(decoded[0], tf.int32),
                truth      = self.trans,
                normalize  = False,
            ))

        elif loss == 'fidelity':
            return tf.reduce_mean(tf.losses.mean_squared_error(
                labels      = self.clean,
                predictions = models['generator']['model'].fidelity,
            ))

        elif loss == 'magnitude':
            return tf.reduce_mean(tf.losses.mean_squared_error(
                labels      = magnitude(self.clean),
                predictions = magnitude(models['generator']['model'].outputs),
            ))
        
        elif loss == 'masking':
            return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                multi_class_labels = self.irm,
                logits             = models['generator']['model'].masking,
            ))

        elif loss == 'speech-mask':
            return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                multi_class_labels = self.ibm_x,
                logits             = models['generator']['model'].outputs,
            ))

        elif loss == 'mimic' or loss == 'map-as-mask-mimic':
            return tf.reduce_mean(tf.losses.mean_squared_error(
                labels      = models['teacher']['model'].outputs,
                predictions = models['student']['model'].outputs,
            ))

        elif loss == 'generator':
            return self.make_adversarial_loss(models)

        else:
            raise ValueError

    def make_adversarial_loss(self, models):
       # Prepare for making the discriminator model
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = self.generator.outputs * epsilon + self.clean * (1-epsilon)

        # Make 3 copies with different inputs
        #self.d_fake = discrimaker(self.generator.outputs)
        #self.d_real = discrimaker(self.clean, reuse=True)
        #d_hat = discrimaker(x_hat, reuse=True)
        #self.feed_dict[d_hat.training] = False

        # Make loss and gradient penalty
        self.discriminator_loss = tf.reduce_mean(self.d_fake.outputs) - tf.reduce_mean(self.d_real.outputs)
        self.discriminator_loss *= 0.005#loss_weight['generator']
        #gradients = tf.gradients(d_hat.outputs, x_hat)[0]
        #slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        #gradient_penalty = 10 * tf.reduce_mean((slopes - 1.0) ** 2)
        #self.discriminator_loss += gradient_penalty

        # Train op
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.discriminator_train_op = self._create_train_op(self.discriminator_loss, var_list)
        self.clip_op = [var.assign(tf.clip_by_value(var, -0.02, 0.02)) for var in var_list]
    
        # Generator loss
        return -tf.reduce_mean(self.d_fake.outputs)

    def _create_train_op(self, loss, var_list, max_norm, learn_rate, decay):
        """ Define the training op """
        
        # Clip gradient
        #grads = tf.gradients(loss, var_list)
        #grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_norm)
        #grad_var_pairs = zip(grads, var_list)

        # Define train op
        global_step = tf.Variable(0, trainable=False)
        #self.learn_rate_pl = tf.placeholder(tf.float32)#
        self.learn_rate_pl = tf.train.exponential_decay(learn_rate, global_step, 1e4, decay)
        opt = tf.train.AdamOptimizer(self.learn_rate_pl)
        #optim = tf.train.GradientDescentOptimizer(self.learn_rate_pl)
        #return optim.apply_gradients(grad_var_pairs, global_step=global_step)
        #return opt.minimize(loss, var_list = var_list, global_step = global_step)


        ## Optimizer definition - nothing different from any classical example
        #opt = tf.train.AdamOptimizer()

        ## Retrieve all trainable variables you defined in your graph
        #tvs = tf.trainable_variables()
        tvs = var_list
        ## Creation of a list of variables with the same shape as the trainable ones
        # initialized with 0s
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        ## Calls the compute_gradients function of the optimizer to obtain... the list of gradients
        gvs = opt.compute_gradients(loss, tvs)

        ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs) if gv[0] is not None]

        ## Define the training step (part with variable value update)
        self.train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])



    def run_ops(self, sess, loader, training = True, epoch = 1):
        """ Run one epoch of batches through the model """

        #if epoch % 4 == 0:
        #    self.learn_rate /= 2

        for var in self.training:
            self.feed_dict[var] = training
        
        # Counters
        start_time = time.time()
        frames = 1
        count = 1

        # Prepare ops
        ops = self.get_ops(training)

        # Iterate dataset
        for batch in loader.batchify(epoch):
            if self.verbose:
                print("Batch", count)
            count += 1

            # Count frames for loss calculation
            frames += batch['frames']

            # Build the feed dict for this batch
            self.build_feed_dict(batch, training, sess, epoch)

            # Run all ops
            output = sess.run(ops, self.feed_dict)

            if training and count % self.batch_size == 0:
                sess.run(self.train_step)
                sess.run(self.zero_ops)

            # Update losses
            for label in self.losses:
                self.losses[label]['loss'] += batch['frames'] * output[self.losses[label]['id']]

                if label == 'generator' and self.verbose:
                    print("G loss", output[self.losses[label]['id']])

            if self.verbose:
                for label in self.losses:
                    print("{}: {}".format(label, self.losses[label]['loss'] / frames))

        # Compute average
        average = {}
        for label in self.losses:
            average[label] = self.losses[label]['loss'] / frames
        duration = time.time() - start_time

        return average, duration

    def get_ops(self, training):

        ops = []
        for i, loss in enumerate(self.losses):
            ops.append(self.losses[loss]['op'])
            self.losses[loss]['id'] = i
            self.losses[loss]['loss'] = 0

        # Doesn't produce output, so no map needed
        if training:
            #ops.append(self.train_op)
            ops.extend(self.accum_ops)

        return ops


    def build_feed_dict(self, batch, training, sess, epoch):

        #self.feed_dict[self.learn_rate_pl] = self.learn_rate

        # Always feed clean, and then either senone or noisy
        if self.clean is not None:
            self.feed_dict[self.clean] = batch['clean']
        if self.senone is not None:
            self.feed_dict[self.senone] = batch['senone']
        if self.noisy is not None:
            self.feed_dict[self.noisy] = batch['noisy']
        if self.irm is not None:
            self.feed_dict[self.irm] = batch['irm']
        if self.ibm_x is not None:
            self.feed_dict[self.ibm_x] = batch['ibm_x']
        if self.ibm_n is not None:
            self.feed_dict[self.ibm_n] = batch['ibm_n']
        if self.trans is not None:
            self.feed_dict[self.trans] = batch['trans']


        # Count the frames in the batch
        if self.lengths:
            self.feed_dict[self.lengths] = [batch['frames']]

        # Train discriminator
        if 'generator' in self.losses:

            self.feed_dict[self.d_real.training] = training
            self.feed_dict[self.d_fake.training] = training
            self.feed_dict[self.generator_train] = False

            d_ops = [self.discriminator_loss, self.discriminator_train_op]
            d_loss, _ = sess.run(d_ops, self.feed_dict)
            sess.run(self.clip_op)

            if self.verbose:
                print("D loss", d_loss)

            self.feed_dict[self.d_real.training] = False
            self.feed_dict[self.d_fake.training] = False
            self.feed_dict[self.generator_train] = training

