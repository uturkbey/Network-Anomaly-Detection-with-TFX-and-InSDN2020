
#This part is mainly adapted from the textbook and reference TFX tutorial

import tensorflow as tf
import tensorflow_transform as tft

#List of feature columns to be used in this project 
#Note: This list also includes the target variable "Label"
FEATURES_TO_BE_USED = ['Src_Port', 'Dst_Port', 'Flow_Duration', 'TotLen_Fwd_Pkts', 
                       'Fwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min', 'Pkt_Len_Max', 'Pkt_Len_Mean', 
                       'Fwd_Pkts/s', 'Bwd_Pkts/s', 'Flow_IAT_Mean', 'Flow_IAT_Min', 'Bwd_IAT_Tot', 
                       'Bwd_IAT_Min', 'Fwd_Header_Len', 'Bwd_Header_Len', 'FIN_Flag_Cnt', 
                       'SYN_Flag_Cnt', 'ACK_Flag_Cnt', 'Init_Bwd_Win_Byts', 'Label']

#First define the numeric feature keys to be normalized in preprocessing function
NUMERIC_FEATURE_KEYS = FEATURES_TO_BE_USED[:20]
LABEL_KEY = FEATURES_TO_BE_USED[20]
NUMBER_OF_CATEGORIES_IN_LABEL_KEY = 5

def preprocessing_fn(inputs):
    
    #  tf.transform's callback function for preprocessing inputs.
    # Args:
    #  inputs: map from feature keys to raw not-yet-transformed features.
    # Returns:
    #  Map from string feature key to transformed feature operations.
    
    #outputs is the dictionary containing the processed data and will be returned at the end of preprocessing function 
    outputs = {}
    
    #Normalize all the numerical features using built in TFT Min-Max Normilizer
#    for key in NUMERIC_FEATURE_KEYS:
#        outputs[transformed_name(key)] = tft.scale_to_0_1(fill_in_missing(inputs[key]))
    
    #Target value transformation
    #This transformation is applied to transform string category names of target key "Label" to 
        #First, indexes of numerical values 0,1,2,3,4 using built in TFT compute_and_apply_vocabulary function. 
            #(For more info about this function, refere to page 71 of "Building Machine Learning Pipelines")
            #Note: top_k parameter is used to make sure that number of categories are limited to 5 including:
            #'DoS/DDoS Attack', 'Malware Attack', 'Normal', 'Other Attack Types', 'Web Attack'
        #And then to one_hot encoded vector representations (of length number of categories)  using user defined
        #convert_num_to_one_hot function
            #See the function belove
    #This transformation is essential in order to apply keras models for multiclass classification 
#Following line executes without an error
#    index = tft.compute_and_apply_vocabulary(fill_in_missing(inputs[LABEL_KEY]), top_k=NUMBER_OF_CATEGORIES_IN_LABEL_KEY) #name->index
#Following line causes an error. Type of index is not approved by cınvert_num_to_one_hot function. Expects a tensor object!!
#    outputs[transformed_name(LABEL_KEY)] = convert_num_to_one_hot(index, num_labels=NUMBER_OF_CATEGORIES_IN_LABEL_KEY) #index->one-hot
 
#---------------------------------------------------------------------------    
    #Alternative for now, it just functions as a buffer for data
    #Either problem with original version above should be solved or transformations should be handled with pandas in csv data 
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = inputs[key]
    outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]
#---------------------------------------------------------------------------
    return outputs

def transformed_name(key):
    #This is a helper function to produce names for processed keys
    #Obtained from "Building Machine Learning Pipelines" page 76
    #Note: key + "_xf" format is conventional and is desired by TFX platform
    return key + '_xf'

def convert_num_to_one_hot(label_tensor, num_labels = 2):
    #This helper function is to convert a given index tensor to a one-hot encoded representation tensor
    #Obtained from "Building Machine Learning Pipelines" page 76
    #Note: this function is very similar to to_categorical function from Keras.util
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape((one_hot_tensor, [-1, num_labels]))

def fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
        x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
    Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(tf.sparse.to_dense(tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]), default_value),axis=1)
