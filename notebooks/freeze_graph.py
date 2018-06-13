
import os, argparse
from resnet_config import ConfigResnet
from tqdm import tqdm
from resnet import model
import tensorflow as tf
config = ConfigResnet()

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    absolute_model_dir = '/'.join(model_dir.split('/')[:-1])
    output_graph = absolute_model_dir + '/frozen_model_Sample.pb.modelzoo'
    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            resnet = model(
                    weight_path=config.ILSVRC12_weight_path,
                    trainable=False,
                    finetune_layers=config.initialize_layers,
                    ILSVRC_activation=config.ILSVRC_activation,
                    finetune_activation=config.finetune_activation,
                    apply_to=config.apply_to,
                    extra_convs=config.extra_convs,
                    squash=config.squash,
                    resnet_size=config.resnet_size)
            train_image = tf.placeholder(tf.float32,
                                shape=[1,224,224,3],
                                name='rgb')
            train_mode = tf.get_variable(
                name='training',
                initializer=False)
            logit = resnet.build(train_image,
                                 training=train_mode,
                                 output_shape=config.output_shape,
                                 feature_attention=config.feature_attention)
        restorer = tf.train.Saver(tf.global_variables())
        # Initialize the graph
        with tf.Session(config=tf.ConfigProto(
                                allow_soft_placement=True
                                            )) as sess:
            # Need to initialize both of these if supplying num_epochs to inputs
            sess.run(tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer()))
            ################## Restore from checkpoint! ##################
            if model_dir is not None:
                restorer.restore(sess, model_dir)
            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
                output_node_names.split(",") # The output node names are used to select the usefull nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

        return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()
    freeze_graph(args.model_dir, args.output_node_names)
