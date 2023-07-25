
#This part is mainly adapted from the reference TFX tutorial

import tensorflow as tf
import tensorflow.keras as keras

import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.tfxio import dataset_options

#List of feature columns to be used in this project 
#Note: This list also includes the target variable "Label"
FEATURES_TO_BE_USED = ['Src_Port', 'Dst_Port', 'Flow_Duration', 'TotLen_Fwd_Pkts', 
                       'Fwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min', 'Pkt_Len_Max', 'Pkt_Len_Mean', 
                       'Fwd_Pkts/s', 'Bwd_Pkts/s', 'Flow_IAT_Mean', 'Flow_IAT_Min', 'Bwd_IAT_Tot', 
                       'Bwd_IAT_Min', 'Fwd_Header_Len', 'Bwd_Header_Len', 'FIN_Flag_Cnt', 
                       'SYN_Flag_Cnt', 'ACK_Flag_Cnt', 'Init_Bwd_Win_Byts', 'Label']

#Numeric feature keys to be used as inputs to model
NUMERIC_FEATURE_KEYS = FEATURES_TO_BE_USED[:20]
#Label key "Label" to be used in model
LABEL_KEY = FEATURES_TO_BE_USED[20]
NUMBER_OF_CATEGORIES_IN_LABEL_KEY = 5 #'DoS/DDoS Attack', 'Malware Attack', 'Normal', 'Other Attack Types', 'Web Attack'


def transformed_name(key):
    return key + '_xf'


def transformed_names(keys):
    return [transformed_name(key) for key in keys]


def get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def build_estimator(config, hidden_units=None, warm_start_from=None):
    """Build an estimator for predicting the tipping behavior of taxi riders.
    Args:
      config: tf.estimator.RunConfig defining the runtime environment for the
        estimator (including model_dir).
      hidden_units: [int], the layer sizes of the DNN (input layer first)
      warm_start_from: Optional directory to warm start from.
    Returns:
      A dict of the following:
        - estimator: The estimator that will be used for training and eval.
        - train_spec: Spec for training.
        - eval_spec: Spec for eval.
        - eval_input_receiver_fn: Input function for eval.
    """
    #??? Note: some features in our dataset are Int values rather than float. They will be converted to float32
    #https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column
    my_feature_columns = [tf.feature_column.numeric_column(key, shape=()) for key in transformed_names(NUMERIC_FEATURE_KEYS)]
  
    #??? Note: Reference TFX tutorial uses DNNLinearCombinedClassifier. Here, DNNClassifier is preferred. 
    #Params arranged accordingly. 
    #https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
    #https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier
    return tf.estimator.DNNClassifier(
        config=config,
        feature_columns=my_feature_columns,
        hidden_units=hidden_units or [100, 70, 50, 25],
        n_classes = NUMBER_OF_CATEGORIES_IN_LABEL_KEY,
        warm_start_from=warm_start_from)


def example_serving_receiver_fn(tf_transform_graph, schema):
    """Build the serving in inputs.
    Args:
      tf_transform_graph: A TFTransformOutput.
      schema: the schema of the input data.
    Returns:
      Tensorflow graph which parses examples, applying tf-transform to them.
    """
    raw_feature_spec = get_raw_feature_spec(schema)
    raw_feature_spec.pop(LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_graph.transform_raw_features(serving_input_receiver.features)

    return tf.estimator.export.ServingInputReceiver(transformed_features, serving_input_receiver.receiver_tensors)



def eval_input_receiver_fn(tf_transform_graph, schema):
    """Build everything needed for the tf-model-analysis to run the model.
    Args:
      tf_transform_graph: A TFTransformOutput.
      schema: the schema of the input data.
    Returns:
      EvalInputReceiver function, which contains:
        - Tensorflow graph which parses raw untransformed features, applies the
          tf-transform preprocessing operators.
        - Set of raw, untransformed features.
        - Label against which predictions will be compared.
    """
    # Notice that the inputs are raw features, not transformed features here.
    raw_feature_spec = get_raw_feature_spec(schema)

    serialized_tf_example = tf.compat.v1.placeholder(dtype=tf.string, shape=[None], name='input_example_tensor')

    # Add a parse_example operator to the tensorflow graph, which will parse
    # raw, untransformed, tf examples.
    features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)

    # Now that we have our raw examples, process them through the tf-transform
    # function computed during the preprocessing step.
    transformed_features = tf_transform_graph.transform_raw_features(features)

    # The key name MUST be 'examples'.
    receiver_tensors = {'examples': serialized_tf_example}

    # NOTE: Model is driven by transformed features (since training works on the
    # materialized output of TFT, but slicing will happen on raw features.
    features.update(transformed_features)

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[transformed_name(LABEL_KEY)])


def input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=200):
    """Generates features and label for tuning/training.
    
    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(batch_size=batch_size, label_key=transformed_name(LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema)


# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
    """Build the estimator using the high level API.
    Args:
      trainer_fn_args: Holds args used to train the model as name/value pairs.
      schema: Holds the schema of the training examples.
    Returns:
      A dict of the following:
        - estimator: The estimator that will be used for training and eval.
        - train_spec: Spec for training.
        - eval_spec: Spec for eval.
        - eval_input_receiver_fn: Input function for eval.
    """
    #Values below are obtained from reference TFX tutorial, might be updated arbitrarily.
    # Number of nodes in the first layer of the DNN
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

    train_batch_size = 40
    eval_batch_size = 40

    tf_transform_graph = tft.TFTransformOutput(trainer_fn_args.transform_output)

    train_input_fn = lambda: input_fn(  # pylint: disable=g-long-lambda
        trainer_fn_args.train_files,
        trainer_fn_args.data_accessor,
        tf_transform_graph,
        batch_size=train_batch_size)

    eval_input_fn = lambda: input_fn(  # pylint: disable=g-long-lambda
        trainer_fn_args.eval_files,
        trainer_fn_args.data_accessor,
        tf_transform_graph,
        batch_size=eval_batch_size)

    train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
        train_input_fn,
        max_steps=trainer_fn_args.train_steps)

    serving_receiver_fn = lambda: example_serving_receiver_fn(  # pylint: disable=g-long-lambda
        tf_transform_graph, schema)

    exporter = tf.estimator.FinalExporter('subset-InSDN', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=trainer_fn_args.eval_steps,
        exporters=[exporter],
        name='subset-InSDN-eval')

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=999, keep_checkpoint_max=1)

    run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)

    estimator = build_estimator(
        # Construct layers sizes with exponetial decay
        hidden_units=[max(2, int(first_dnn_layer_size * dnn_decay_factor**i)) for i in range(num_dnn_layers)],
        config=run_config,
        warm_start_from=trainer_fn_args.base_model) #??? Is there always a base model? 

    # Create an input receiver for TFMA processing
    receiver_fn = lambda: eval_input_receiver_fn(  # pylint: disable=g-long-lambda
        tf_transform_graph, schema)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': receiver_fn}
