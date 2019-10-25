import numpy as np
import tensorflow as tf
from math import acos
####################################################################################
##################################### CONFIG #######################################
class Config(object):
    ###============ player params =========
    CAP_RANGE = 2
    VD = 1.
    VI = 1.5
    TIME_STEP = 0.09
    

    LB = 2*acos(VD/VI)

    X0s = [np.array([0,-6.])*CAP_RANGE, np.array([0,6.])*CAP_RANGE, np.array([10., 0.])*CAP_RANGE]

    ###============ learning params =========
    LEARNING_RATE = 0.01
    LAYER_SIZES = [30, 6, 30]
    ACT_FUNCS = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh]
    TAU = 0.01
    MAX_BUFFER_SIZE = 10000
    BATCH_SIZE = 1000
    FIT_STEPS = 88
    TRAIN_STEPS = 100
    TARGET_UPDATE_INTERVAL = 1

    ###============ saving params =========
    DATA_FILE = 'valueData.csv'
    MODEL_DIR = 'models/'
    MODEL_FILE = 'valueFn'

    SAVE_FREQUENCY = 100
    PRINTING_FREQUENCY = 50
