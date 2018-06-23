from os import listdir, rename, getcwd
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import time
import sys
from PIL import Image
import requests
from io import BytesIO
from data.load_data import load_data


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file(file_vector,
                                input_height=224,
                                input_width=224,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"

    dims_expander = tf.expand_dims(file_vector, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def is_obstacle(label):
    ground_labels = ['grass', 'sidewalk', 'street']
    if label in ground_labels:
        return False
    else:
        return True


if __name__ == "__main__":
    currentDir = getcwd()
    modelFile = join(currentDir, "output_graph.pb")
    label_file = join(currentDir, "output_labels.txt")
    input_layer = "input"
    output_layer = "final_result"

    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(modelFile, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    data = load_data(
        '/Users/raulpopovici/Desktop/licenta/bt-tf-training/tests')
    labels = load_labels(label_file)
    wrong = 0
    count = 0
    missmatch = 0
    false_positive = 0
    false_negative = 0
    with tf.Session(graph=graph) as sess:
        for label in data:
            if label not in labels:
                raise Exception('wrong label name ' + label)

            for img in data[label]:
                count = count + 1
                imageTensor = read_tensor_from_image_file(img)
                # start = time.time()
                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: imageTensor})
                results = np.squeeze(results)
                top_k = results.argsort()[-5:][::-1]
                top_1 = labels[top_k[0]]
                if top_1 != label:
                    wrong = wrong + 1
                    print(wrong, label, top_1)
                    if is_obstacle(top_1) != is_obstacle(label):
                        missmatch = missmatch + 1
                        if is_obstacle(top_1):
                            false_positive = false_positive + 1
                        else:
                            false_negative = false_negative + 1
                        print('missmatch obstacle', missmatch)

            print('done', label)

    print(wrong, count)
    print('bad accuracy', wrong/count)
    print(missmatch, count)
    print('missmath obstacle', 1 - missmatch/count)
    print('false negatives', false_negative, false_negative / count)
    print('false positives', false_positive, false_positive / count)
