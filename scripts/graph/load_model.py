import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from data.load_data import load_data

CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'


def load_model(model_dir, model_info):
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(
            model_dir, model_info['model_file_name'])
        print('Model path: ', model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            message = f.read()

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(message)
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


def add_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_layer(class_count, final_tensor_name, bottleneck_tensor,
                    bottleneck_tensor_size, quantize_layer, is_training, learning_rate):
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(
            tf.int64, [None], name='GroundTruthInput')

    with tf.name_scope('final_retrain_ops'):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(
                tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def build_eval_session(model_info, class_count, learning_rate):
    eval_graph, bottleneck_tensor, _ = create_model_graph(model_info)
    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        (_, _, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, FLAGS.final_tensor_name, bottleneck_tensor,
            model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
            False, learning_rate)

        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

        evaluation_step, prediction = add_evaluation_step(final_tensor,
                                                          ground_truth_input)

    return (eval_sess, bottleneck_input, ground_truth_input, evaluation_step,
            prediction)


def save_graph_to_file(graph, graph_file_name, model_info, class_count):
    sess, _, _, _, _ = build_eval_session(model_info, class_count)
    graph = sess.graph

    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return


def create_model_info(architecture):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
        tf.logging.error("Couldn't understand architecture name '%s'",
                         architecture)
        return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
            version_string != '0.5' and version_string != '0.25'):
        tf.logging.error(
            """"The Mobilenet version should be '1.0', '0.75', '0.5', or '0.25',
    but found '%s' for architecture '%s'""", version_string, architecture)
        return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
            size_string != '160' and size_string != '128'):
        tf.logging.error(
            """The Mobilenet input size should be '224', '192', '160', or '128',
    but found '%s' for architecture '%s'""",
            size_string, architecture)
        return None
    if len(parts) == 3:
        is_quantized = False
    else:
        if parts[3] != 'quant':
        tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
            architecture)
        return None
        is_quantized = True

    data_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/'
    model_name = 'mobilenet_v1_' + version_string + '_' + size_string
    if is_quantized:
        model_name += '_quant'
    data_url += model_name + '.tgz'
    bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
    resized_input_tensor_name = 'input:0'
    model_file_name = model_name + '_frozen.pb'

    bottleneck_tensor_size = 1001
    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    input_mean = 127.5
    input_std = 127.5
    else:
        tf.logging.error(
            "Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'quantize_layer': is_quantized,
    }


def export_model(model_info, class_count, saved_model_dir):
    sess, _, _, _, _ = build_eval_session(model_info, class_count)
    graph = sess.graph
    with graph.as_default():
        input_tensor = model_info['resized_input_tensor_name']
        in_image = sess.graph.get_tensor_by_name(input_tensor)
        inputs = {'image': tf.saved_model.utils.build_tensor_info(in_image)}

        out_classes = sess.graph.get_tensor_by_name('final_result:0')
        outputs = {
            'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')

        # Save out the SavedModel.
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            },
            legacy_init_op=legacy_init_op)
        builder.save()
