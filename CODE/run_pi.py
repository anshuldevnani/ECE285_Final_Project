from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
import time
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model")
args = parser.parse_args()

co = {}
_add_supported_quantized_objects(co)
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

model = tf.keras.models.load_model(str(args.model) + '.h5', custom_objects=co)
qmodel = strip_pruning(model)

X_test_reduced = np.load('x_test.npy')

start_time = time.time()
y_predict_q = qmodel.predict(X_test_reduced)
elapsed_time = (time.time() - start_time)
print("Latency - {}".format(elapsed_time))
print("Throughput - {} images / sec".format(1500 / float(elapsed_time)))