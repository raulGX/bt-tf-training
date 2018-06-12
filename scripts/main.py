from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

FLAGS = argparse.Namespace(architecture='mobilenet_1.0_224',
                           bottleneck_dir='/tmp/bottleneck',
                           eval_step_interval=50,
                           final_tensor_name='final_result',
                           flip_left_right=True,
                           how_many_training_steps=1000,
                           image_dir='download-dataset/downloads/',
                           intermediate_output_graphs_dir='/tmp/intermediate_graph/',
                           intermediate_store_frequency=0,
                           learning_rate=0.001, model_dir='/tmp/imagenet',
                           output_graph='/tmp/output_graph.pb',
                           output_labels='/tmp/output_labels.txt',
                           print_misclassified_test_images=False,
                           random_brightness=30,
                           random_crop=0,
                           random_scale=30,
                           saved_model_dir='/tmp/saved_models/1/',
                           summaries_dir='/tmp/retrain_logs',
                           test_batch_size=-1,
                           testing_percentage=10,
                           train_batch_size=32,
                           validation_batch_size=-1,
                           validation_percentage=10)

from graph.retrain import *
set_flags(FLAGS)

# Needed to make sure the logging output is visible.
# See https://github.com/tensorflow/tensorflow/issues/3047
tf.logging.set_verbosity(tf.logging.INFO)

# Prepare necessary directories that can be used during training
prepare_file_system()

# Gather information about the model architecture we'll be using.
model_info = create_model_info(FLAGS.architecture)

if not model_info:
    tf.logging.error('Did not recognize architecture flag')
    raise Exception('Error')

# Look at the folder structure, and create lists of all the images.
image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                 FLAGS.validation_percentage)
class_count = len(image_lists.keys())
if class_count == 0:
    tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
    raise Exception('Error')
if class_count == 1:
    tf.logging.error('Only one valid folder of images found at ' +
                     FLAGS.image_dir +
                     ' - multiple classes are needed for classification.')

# See if the command-line flags mean we're applying any distortions.
do_distort_images = should_distort_images(
    FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
    FLAGS.random_brightness)
print(do_distort_images)
# Set up the pre-trained graph.
maybe_download_and_extract(model_info['data_url'])
graph, bottleneck_tensor, resized_image_tensor = (
    create_model_graph(model_info))

# Add the new layer that we'll be training.
with graph.as_default():
    (train_step, cross_entropy, bottleneck_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, FLAGS.final_tensor_name, bottleneck_tensor,
         model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
         True)

with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    if do_distort_images:
        # We will be applying distortions, so setup the operations we'll need.
        (distorted_jpeg_data_tensor,
         distorted_image_tensor) = add_input_distortions(
             FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
             FLAGS.random_brightness, model_info['input_width'],
             model_info['input_height'], model_info['input_depth'],
             model_info['input_mean'], model_info['input_std'])
    else:
        # We'll make sure we've calculated the 'bottleneck' image summaries and
        # cached them on disk.
        cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                          FLAGS.bottleneck_dir, jpeg_data_tensor,
                          decoded_image_tensor, resized_image_tensor,
                          bottleneck_tensor, FLAGS.architecture)

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    # Create a train saver that is used to restore values into an eval graph
    # when exporting models.
    train_saver = tf.train.Saver()

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the training for as many cycles as requested on the command line.
    for i in range(FLAGS.how_many_training_steps):
        # Get a batch of input bottleneck values, either calculated fresh every
        # time with distortions applied, or from the cache stored on disk.
        if do_distort_images:
            (train_bottlenecks,
             train_ground_truth) = get_random_distorted_bottlenecks(
                 sess, image_lists, FLAGS.train_batch_size, 'training',
                 FLAGS.image_dir, distorted_jpeg_data_tensor,
                 distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
        else:
            (train_bottlenecks,
             train_ground_truth, _) = get_random_cached_bottlenecks(
                 sess, image_lists, FLAGS.train_batch_size, 'training',
                 FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                 decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                 FLAGS.architecture)
        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Capture training summaries for TensorBoard with the `merged` op.
        train_summary, _ = sess.run(
            [merged, train_step],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        train_writer.add_summary(train_summary, i)

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                            (datetime.now(), i, train_accuracy * 100))
            tf.logging.info('%s: Step %d: Cross entropy = %f' %
                            (datetime.now(), i, cross_entropy_value))

            validation_bottlenecks, validation_ground_truth, _ = (
                get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.validation_batch_size, 'validation',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    FLAGS.architecture))
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step],
                feed_dict={bottleneck_input: validation_bottlenecks,
                           ground_truth_input: validation_ground_truth})
            validation_writer.add_summary(validation_summary, i)
            tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (datetime.now(), i, validation_accuracy * 100,
                             len(validation_bottlenecks)))

        # Store intermediate results
        intermediate_frequency = FLAGS.intermediate_store_frequency

        if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                and i > 0):
            # If we want to do an intermediate save, save a checkpoint of the train
            # graph, to restore into the eval graph.
            train_saver.save(sess, CHECKPOINT_NAME)
            intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                      'intermediate_' + str(i) + '.pb')
            tf.logging.info('Save intermediate result to : ' +
                            intermediate_file_name)
            save_graph_to_file(graph, intermediate_file_name, model_info,
                               class_count)

    # After training is complete, force one last save of the train checkpoint.
    train_saver.save(sess, CHECKPOINT_NAME)

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    run_final_eval(sess, model_info, class_count, image_lists, jpeg_data_tensor,
                   decoded_image_tensor, resized_image_tensor,
                   bottleneck_tensor)

    # Write out the trained graph and labels with the weights stored as
    # constants.
    save_graph_to_file(graph, FLAGS.output_graph, model_info, class_count)
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')

    export_model(model_info, class_count, FLAGS.saved_model_dir)
