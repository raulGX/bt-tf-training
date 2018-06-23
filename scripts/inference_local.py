from os import listdir, rename, getcwd
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import time
import sys
from PIL import Image
import requests
from io import BytesIO
from scipy import misc


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file(image,
                                input_height=224,
                                input_width=224,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"

    dims_expander = tf.expand_dims(image, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


if __name__ == "__main__":
    path = sys.argv[1]
    img = misc.imread(path)
    img = img.astype(float)
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
    with tf.Session(graph=graph) as sess:
        imageTensor = read_tensor_from_image_file(img)
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: imageTensor})
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        for i in top_k:
            print(labels[i], results[i])
        end = time.time()
        print(end-start)
