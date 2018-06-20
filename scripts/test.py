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

    # image_reader = tf.image.decode_image(file_vector)
    # float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(file_vector, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


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
                if labels[top_k[0]] != label:
                    wrong = wrong + 1
                    print(wrong, label, labels[top_k[0]])

            print('done', label)

    print(wrong, count)
    print('bad accuracy', wrong/count)
