from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json

from resnet import ResNet
from data_io import DataLoader
from trainer import Trainer

parser = argparse.ArgumentParser()

# Files
parser.add_argument("--base_directory", default=None, help="The directory the data is in")
parser.add_argument("--numpy_flist", default=None, help="The directory the data is in")
parser.add_argument("--clean_flist", default=None, help="The directory the data is in")
parser.add_argument("--noise_flist", default=None, help="The directory the data is in")
parser.add_argument("--noisy_flist", default=None, help="The directory the data is in")
parser.add_argument("--clean_scp", default=None, help="The directory the data is in")
parser.add_argument("--noisy_scp", default=None, help="The directory the data is in")
parser.add_argument("--senone_file", default=None, help="The directory the data is in")
parser.add_argument("--trans_file", default=None, help="The directory the data is in")
parser.add_argument("--teacher_pretrain", default=None, help="directory with critic weights")
parser.add_argument("--student_pretrain", default=None, help="directory with critic weights")
parser.add_argument("--generator_pretrain", default=None, help="directory with generator pretrained weights")
parser.add_argument("--generator_checkpoints", default=None, help="directory to store generator weights")
parser.add_argument("--student_checkpoints", default=None, help="directory to store student weights")
parser.add_argument("--student_model", default="resnet", help="resnet or lstm or dnn")
#parser.add_argument("--model_file", default=None)

# Training
parser.add_argument("--learn_rate", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--lr_decay", type=float, default=0.95)

# Model
parser.add_argument("--glayers", type=int, default=2)
parser.add_argument("--gunits", type=int, default=2048)
parser.add_argument("--gfilters", type=int, nargs="+", default=[128, 128, 256, 256])
parser.add_argument("--tlayers", type=int, default=2)
parser.add_argument("--tunits", type=int, default=2048)
parser.add_argument("--tfilters", type=int, nargs="+", default=[128, 128, 256, 256])
parser.add_argument("--slayers", type=int, default=2)
parser.add_argument("--sunits", type=int, default=2048)
parser.add_argument("--sfilters", type=int, nargs="+", default=[128, 128, 256, 256])
parser.add_argument("--dropout", type=float, default=0.3, help="percentage of neurons to drop")

# Data
parser.add_argument("--input_featdim", type=int, default=256)
parser.add_argument("--output_featdim", type=int, default=256)
parser.add_argument("--senones", type=int, default=2007)
parser.add_argument("--characters", type=int, default=27)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
#parser.add_argument("--noise_mask", default=False, action="store_true")

# Loss weights
parser.add_argument("--loss_weight", type=json.loads, default={'fidelity': 1.0})

a = parser.parse_args()

def run_training():
    """ Define our model and train it """

    # Create models if its been pretrained, or we're training it
    load_generator = a.generator_pretrain is not None
    train_generator = a.generator_checkpoints is not None

    load_teacher = a.teacher_pretrain is not None

    load_student = a.student_pretrain is not None
    train_student = a.student_checkpoints is not None

    models = {}

    with tf.Graph().as_default():

        # Define our generator model
        if load_generator or train_generator:
            with tf.variable_scope('generator'):
                noisy_inputs = tf.placeholder(tf.float32, [None, a.channels, None, a.input_featdim], name='noisy')
                generator = ResNet(
                    inputs      = noisy_inputs,
                    output_dim  = a.output_featdim,
                    output_type = a.loss_weight.keys() & ['fidelity', 'masking', 'map-as-mask-mimic'],
                    fc_nodes    = a.gunits,
                    fc_layers   = a.glayers,
                    filters     = a.gfilters,
                    dropout     = a.dropout,
                )
            generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            load_generator_vars = [var for var in generator_vars if '_scale' not in var.op.name and '_shift' not in var.op.name]
            generator_loader = tf.train.Saver(load_generator_vars)
            generator_saver = tf.train.Saver(generator_vars)
            models['generator'] = {'model': generator, 'train': train_generator, 'vars': generator_vars}

        if load_teacher:
            with tf.variable_scope('teacher'):
                clean_inputs = tf.placeholder(tf.float32, [None, a.channels, None, a.output_featdim], name='clean')
                teacher = ResNet(
                    inputs     = clean_inputs,
                    output_dim = a.senones,
                    fc_nodes   = a.tunits,
                    fc_layers  = a.tlayers,
                    filters    = a.tfilters,
                    dropout    = 0,
                )
            teacher_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='teacher')
            teacher_saver = tf.train.Saver({'mimic' + var.op.name[7:]: var for var in teacher_vars})
            models['teacher'] = {'model': teacher, 'train': False, 'vars': teacher_vars}

        # Define critic for generating outputs
        if load_student or train_student:
            if load_generator or train_generator:
                inputs = generator.outputs
                #noisy_min = tf.reduce_min(noisy_inputs)
                #inputs = tf.multiply(tf.sigmoid(generator.outputs), noisy_inputs - noisy_min) + noisy_min
            else:
                inputs = tf.placeholder(tf.float32, [None, a.channels, None, a.input_featdim], name='clean')

            with tf.variable_scope('mimic'):
                student =  ResNet(
                    inputs     = inputs,
                    output_dim = a.senones,
                    fc_nodes   = a.sunits,
                    fc_layers  = a.slayers,
                    filters    = a.sfilters,
                    dropout    = a.dropout,
                )
            student_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mimic')
            student_saver = tf.train.Saver(student_vars)
            models['student'] = {'model': student, 'train': train_student, 'vars': student_vars}

        flists = []
        for flist in [
            ('clean', 'json', a.clean_flist),
            ('noisy', 'json', a.noisy_flist),
            ('noise', 'json', a.noise_flist),
            ('numpy', 'json', a.numpy_flist),
            ('clean', 'scp', a.clean_scp),
            ('noisy', 'scp', a.noisy_scp),
            ('senone', 'txt', a.senone_file),
            ('trans', 'txt', a.trans_file),
        ]:
            if flist[-1] is not None:
                flists.append(flist)

        # Create loader for train data
        train_loader = DataLoader(
            base_dir    = a.base_directory,
            flists      = flists,
            stage       = 'tr',
            shuffle     = False,
            channels    = a.channels,
            compute_irm = 'masking' in a.loss_weight,
            #noise_mask  = a.noise_mask,
            #logify      = True,
            batch_size  = a.batch_size,
        )

        # Create loader
        dev_loader = DataLoader(
            base_dir    = a.base_directory,
            flists      = flists,
            stage       = 'dt',
            shuffle     = False,
            channels    = a.channels,
            compute_irm = 'masking' in a.loss_weight,
            #noise_mask  = a.noise_mask,
            #logify      = True,
        )

        trainer = Trainer(
            models      = models,
            learn_rate  = a.learn_rate,
            lr_decay    = a.lr_decay,
            loss_weight = a.loss_weight,
        )

        # Begin session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Load critic weights, generator weights and initialize trainer weights
        #if a.generator_pretrain and a.model_file:
        #    generator_saver.restore(sess, os.path.join(a.generator_pretrain, a.model_file))

        if a.generator_pretrain:
            sess.run(tf.variables_initializer(generator.scale_vars))
            generator_loader.restore(sess, tf.train.latest_checkpoint(a.generator_pretrain))
        else:
            sess.run(tf.variables_initializer(generator_vars))

        # Load teacher
        if a.teacher_pretrain:
            teacher_saver.restore(sess, tf.train.latest_checkpoint(a.teacher_pretrain))
        
        # Load student
        if a.student_pretrain:
            student_saver.restore(sess, tf.train.latest_checkpoint(a.student_pretrain))
        elif train_student:
            sess.run(tf.variables_initializer(student_vars))

        # Perform training
        min_loss = float('inf')
        for epoch in range(1, 200):
            print('Epoch %d' % epoch)

            # Run train ops
            losses, duration = trainer.run_ops(sess, train_loader, training = True, epoch = epoch)
            for loss in a.loss_weight:
                print('{} loss: {:.6f}'.format(loss, losses[loss]))
            print('Train loss: %.6f (%.3f sec)' % (losses['average'], duration))

            # Run eval ops
            losses, duration = trainer.run_ops(sess, dev_loader, training = False)
            eval_loss = losses['average']
            for loss in a.loss_weight:
                print('{} loss: {:.6f}'.format(loss, losses[loss]))
            print('Eval loss: %.6f (%.3f sec)\n' % (eval_loss, duration))

            if 'cer' in losses:
                eval_loss = losses['cer']

            # Save if we've got the best loss so far
            if eval_loss < min_loss:
                min_loss = eval_loss
                if a.generator_checkpoints:
                    save_file = os.path.join(a.generator_checkpoints, "model-{0:.4f}.ckpt".format(eval_loss))
                    save_path = generator_saver.save(sess, save_file, global_step = epoch)

                if a.student_checkpoints:
                    save_file = os.path.join(a.student_checkpoints, "model-{0:.4f}.ckpt".format(eval_loss))
                    save_path = student_saver.save(sess, save_file, global_step = epoch)

def main():
    run_training()

if __name__=='__main__':
    main()

