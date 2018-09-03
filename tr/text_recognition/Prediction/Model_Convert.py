from tensorflow.python.tools.freeze_graph import freeze_graph
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import tfcoreml


def load_graph(frozen_graph_path):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_path, "rb") as fp:
        graph_def = tf.GraphDef();
        graph_def.ParseFromString(fp.read());

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph;


def display_node(graph_def):
    nodes = graph_def.node
    N = len(nodes)
    for i in range(N):
        print('\n\nop id {} : node name: "{}"'.format(str(i), nodes[i].name))
        print('input(s):')
        for x in nodes[i].input:
            print("name = {}, ".format(x))


def display_op(graph):
    ops = graph.get_operations()
    N = len(ops)
    for i in range(N):
        print('\n\nop id {} : op name: "{}"'.format(str(i), ops[i].name))
        print('input(s):')
        for x in ops[i].inputs:
            print("name = {}, shape: {}, ".format(x.name, x.get_shape()))
        print('\noutput(s):')
        for x in ops[i].outputs:
            print("name = {}, shape: {},".format(x.name, x.get_shape()))


def remove_dropout_node(graph, output_op_name, input_op_name, dst_node_name_list):
    output_op = graph.get_operation_by_name(output_op_name);
    input_op = graph.get_operation_by_name(input_op_name);

    ge.connect(output_op, input_op);
    graph_def = graph.as_graph_def();
    # display_node(graph_def);
    new_graph_def = tf.graph_util.extract_sub_graph(graph_def, dst_node_name_list);

    return new_graph_def;


def freeze_model(graph_def_path, checkpoint_path, frozen_model_path, output_node_name_list, remove_output_op_name, remove_input_op_name):
    """
        Step 1: "Freeze" your tensorflow model - convert your TF model into a stand-alone graph definition file
        Inputs:
        (1) TensorFlow code
        (2) trained weights in a checkpoint file
        (3) The output tensors' name you want to use in inference
        (4) [Optional] Input tensors' name to TF model
        Outputs:
        (1) A frozen TensorFlow GraphDef, with trained weights frozen into it
        """
    output_node_names = ",".join(output_node_name_list);

    # Call freeze graph
    freeze_graph(input_graph=graph_def_path,
                 input_saver="",
                 input_binary=False,
                 input_checkpoint=checkpoint_path,
                 output_node_names=output_node_names,
                 restore_op_name="save/restore_all",
                 filename_tensor_name="save/Const:0",
                 output_graph=frozen_model_path,
                 clear_devices=True,
                 initializer_nodes="");

    graph = load_graph(frozen_model_path);
    display_op(graph);
    new_graph_def = remove_dropout_node(graph, remove_output_op_name, remove_input_op_name, output_node_name_list);

    with tf.gfile.GFile(frozen_model_path, 'wb') as fp:
        fp.write(new_graph_def.SerializeToString())


def tf2coreml(frozen_model_path, coreml_model_path, input_tensor_shapes, output_tensor_names):

    """
    Step 2: Call converter
    """
    # Provide these inputs in addition to inputs in Step 1
    # A dictionary of input tensors' name and shape (with batch)

    # Call the converter
    coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_path,
        mlmodel_path=coreml_model_path,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names);

    return coreml_model;