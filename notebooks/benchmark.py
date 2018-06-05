from os import listdir, rename, getcwd
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import time
def read_tensor_from_image_file(file_name,
                                input_height=224,
                                input_width=224,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result
if __name__ == "__main__":
    currentDir = getcwd()
    imagesDir = currentDir + "/images"
    modelFile = join(currentDir, "graph_optimized.pb")
    input_layer = "input"
    output_layer = "final_result"

    images = [join(imagesDir, f) for f in listdir(imagesDir) if isfile(join(imagesDir, f))] 
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
        start = time.time()
        imageTensor = read_tensor_from_image_file(images[0])
        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: imageTensor})
        end=time.time()
        print(end-start)
    with tf.Session(graph=graph) as sess:
        start = time.time()
        for image in images:
            imageTensor = read_tensor_from_image_file(image)
            results = sess.run(output_operation.outputs[0],
                            {input_operation.outputs[0]: imageTensor})
        end=time.time()
        print(end-start)
    print(len(images))