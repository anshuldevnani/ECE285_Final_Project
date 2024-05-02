from axi_stream_driver import NeuralNetworkOverlay
import numpy as np

X_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

nn = NeuralNetworkOverlay('hls4ml_nn.bit', X_test.shape, y_test.shape)
print("ON HARDWARE")
y_hw, latency, throughput = nn.predict(X_test, profile=True)
print("Latency - {}".format(latency))
np.save('y_hw.npy',y_hw)
