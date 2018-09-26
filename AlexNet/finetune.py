"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf
import dataread as dr

from AlexNet.alexnet import AlexNet
from datetime import datetime

"""
Configuration Part.
"""

################################################################
train, validation, train_labels, validation_labels = dr.read(False)
batch_szie = 64
batch_i = 0
def get_next_batch(batch_size=64):
    global batch_i

    if (batch_i + 1) * batch_size > len(train):
        end = len(train)
        batch_i = -1
    else:
        end = (batch_i + 1) * batch_size
    batch_x = train[batch_i * batch_size : end]
    batch_y = train_labels[batch_i * batch_size: end]

    batch_i = batch_i + 1
    return batch_x, batch_y
####################################################################

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"
checkpoint_path = "/tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, dr.ROWS, dr.COLS, dr.CHANNELS])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
# Start Tensorflow session

# 用于测试的变换
crop = tf.random_crop(x, [49, 192, 192, 3])
crop = tf.image.resize_images(crop, [256, 256])
lr_op = tf.image.flip_left_right(x)
ud_op = tf.image.flip_up_down(x)
lrud_op = tf.image.transpose_image(x)

with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        for step in range(len(train) // batch_szie):

            # get next batch of data
            img_batch, label_batch = get_next_batch(batch_szie)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, len(train) // batch_szie + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))

        # -------test———————————#
        test_score = sess.run(score, feed_dict={x: validation,
                                            y: validation_labels,
                                                keep_prob: 1.})
        # crop
        crop_batch = crop.eval(feed_dict = {x : validation})
        test_score += sess.run(score, feed_dict={x: crop_batch,
                                            y: validation_labels,
                                                keep_prob: 1.})
        # flip_lr
        lr_op_batch = lr_op.eval(feed_dict={x: validation})
        test_score += sess.run(score, feed_dict={x: lr_op_batch,
                                                 y: validation_labels,
                                                 keep_prob: 1.})
        # flip_up
        ud_op_batch = lr_op.eval(feed_dict={x: validation})
        test_score += sess.run(score, feed_dict={x: ud_op_batch,
                                                 y: validation_labels,
                                                 keep_prob: 1.})

        # flip_lrud
        lrud_op_batch = lr_op.eval(feed_dict={x: validation})
        test_score += sess.run(score, feed_dict={x: lrud_op_batch,
                                                 y: validation_labels,
                                                 keep_prob: 1.})

        test_score = test_score / 5
        test_pred = tf.equal(tf.argmax(test_score, 1), tf.argmax(y, 1))
        test_acc = tf.reduce_mean(tf.cast(test_pred, tf.float32)).eval()
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(test_acc))
        # -------end———————————#

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
