import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from data.load_data import load_data
from graph.load_model import load_model


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.abspath(os.path.join(
        dir_path, 'model/downloads'))
    model_info = {
        'model_file_name': 'mobilenet_v1_1.0_224_frozen.pb',
        'bottleneck_tensor_name': 'MobilenetV1/Predictions/Reshape:0',
        'resized_input_tensor_name': 'input:0'}
    graph, bottleneck_tensor, resized_input_tensor = load_model(
        model_path, model_info)
    img_paths = os.path.abspath(os.path.join(
        dir_path, 'download-dataset/downloads'))
    images = load_data(img_paths)

    with tf.Session(graph=graph) as sess:

        imageTensor = images['car'][0]
        results = sess.run(bottleneck_tensor.outputs[0],
                           {resized_input_tensor.outputs[0]: imageTensor})
        print(results)


if __name__ == '__main__':
    main()
