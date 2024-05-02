from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# {'CSynthesisReport': {'TargetClockPeriod': '10.00',
#   'EstimatedClockPeriod': '8.750',
#   'BestLatency': '38710',
#   'WorstLatency': '38971',
#   'IntervalMin': '26627',
#   'IntervalMax': '38926',
#   'BRAM_18K': '107',
#   'DSP': '148',
#   'FF': '56780',
#   'LUT': '92250',
#   'URAM': '0',
#   'AvailableBRAM_18K': '280',
#   'AvailableDSP': '220',
#   'AvailableFF': '106400',
#   'AvailableLUT': '53200',
#   'AvailableURAM': '0'},
#  'TimingReport': {'WNS': -3.698,
#   'TNS': -11719.172,
#   'WHS': 0.009,
#   'THS': 0.0,
#   'WPWS': 3.75,
#   'TPWS': 0.0}}

# ON HARDWARE
# Classified 1500 samples in 0.574743 seconds (2609.862147081391 inferences / s)
# Latency - 0.574743

# Latency - 3.5589351654052734
# Throughput - 421.47438216374127 images / sec

def model1():

    input_shape = (32,32,3)
    x = x_in = Input(shape=input_shape)

###### 0
    x = QConv2DBatchnorm(
            8,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_quantizer="quantized_bits(12,4,alpha=1)",
            bias_quantizer="quantized_bits(12,4,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,
            name='convbn_{}'.format(0),
        )(x)

    x = QActivation('quantized_relu(12)', name='relu_conv_%i' % 0)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(0))(x)

###### 1
    x = QConv2DBatchnorm(
        8,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(12,4,alpha=1)",
        bias_quantizer="quantized_bits(12,4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(1),
    )(x)

    x = QActivation('quantized_relu(12)', name='relu_conv_%i' % 1)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(1))(x)

######## 2
    x = QConv2DBatchnorm(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(12,4,alpha=1)",
        bias_quantizer="quantized_bits(12,4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(2),
    )(x)

    x = QActivation('quantized_relu(12)', name='relu_conv_%i' % 2)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(2))(x)


    x = Flatten()(x)

########## 0
    x = QDense(
        42,
        kernel_quantizer="quantized_bits(12,4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % 0,
        use_bias=False,
    )(x)

    x = BatchNormalization(name='bn_dense_{}'.format(0))(x)

    x = QActivation('quantized_relu(12)', name='relu_dense_%i' % 0)(x)

########## 1
    x = QDense(
        64,
        kernel_quantizer="quantized_bits(12,4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % 1,
        use_bias=False,
    )(x)

    x = BatchNormalization(name='bn_dense_{}'.format(1))(x)

    x = QActivation('quantized_relu(12)', name='relu_dense_%i' % 1)(x)

    x = Dense(int(43), name='output_dense')(x)
    x_out = Activation('softmax', name='output_softmax')(x)
    qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
    return qmodel





# Change number of filters
# {'CSynthesisReport': {'TargetClockPeriod': '10.00',
#   'EstimatedClockPeriod': '8.750',
#   'BestLatency': '24681',
#   'WorstLatency': '24814',
#   'IntervalMin': '24579',
#   'IntervalMax': '24579',
#   'BRAM_18K': '63',
#   'DSP': '105',
#   'FF': '44597',
#   'LUT': '78441',
#   'URAM': '0',
#   'AvailableBRAM_18K': '280',
#   'AvailableDSP': '220',
#   'AvailableFF': '106400',
#   'AvailableLUT': '53200',
#   'AvailableURAM': '0'},
#  'TimingReport': {'WNS': -1.652,
#   'TNS': -3259.327,
#   'WHS': 0.018,
#   'THS': 0.0,
#   'WPWS': 3.75,
#   'TPWS': 0.0}}

# ON HARDWARE
# Classified 1500 samples in 0.42099699999999995 seconds (3562.9707575113366 inferences / s)
# Latency - 0.42099699999999995

# Latency - 3.20943284034729
# Throughput - 467.37229741741106 images / sec

# Accuracy QKeras, CPU:     0.8073333333333333
# Accuracy hls4ml, pynq-z2: 0.7953333333333333

def model2():

    input_shape = (32,32,3)
    x = x_in = Input(shape=input_shape)

###### 0
    x = QConv2DBatchnorm(
            2,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_quantizer="quantized_bits(12,4,alpha=1)",
            bias_quantizer="quantized_bits(12,4,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,
            name='convbn_{}'.format(0),
        )(x)

    x = QActivation('quantized_relu(12)', name='relu_conv_%i' % 0)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(0))(x)

###### 1
    x = QConv2DBatchnorm(
        4,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(12,4,alpha=1)",
        bias_quantizer="quantized_bits(12,4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(1),
    )(x)

    x = QActivation('quantized_relu(12)', name='relu_conv_%i' % 1)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(1))(x)

######## 2
    x = QConv2DBatchnorm(
        8,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(12,4,alpha=1)",
        bias_quantizer="quantized_bits(12,4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(2),
    )(x)

    x = QActivation('quantized_relu(12)', name='relu_conv_%i' % 2)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(2))(x)


    x = Flatten()(x)

########## 0
    x = QDense(
        42,
        kernel_quantizer="quantized_bits(12,4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % 0,
        use_bias=False,
    )(x)

    x = BatchNormalization(name='bn_dense_{}'.format(0))(x)

    x = QActivation('quantized_relu(12)', name='relu_dense_%i' % 0)(x)

########## 1
    x = QDense(
        64,
        kernel_quantizer="quantized_bits(12,4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % 1,
        use_bias=False,
    )(x)

    x = BatchNormalization(name='bn_dense_{}'.format(1))(x)

    x = QActivation('quantized_relu(12)', name='relu_dense_%i' % 1)(x)

    x = Dense(int(43), name='output_dense')(x)
    x_out = Activation('softmax', name='output_softmax')(x)
    qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
    return qmodel


# Change quantization to 4 bit
# {'CSynthesisReport': {'TargetClockPeriod': '10.00',
#   'EstimatedClockPeriod': '8.750',
#   'BestLatency': '38465',
#   'WorstLatency': '38725',
#   'IntervalMin': '22051',
#   'IntervalMax': '38701',
#   'BRAM_18K': '84',
#   'DSP': '3',
#   'FF': '48244',
#   'LUT': '97045',
#   'URAM': '0',
#   'AvailableBRAM_18K': '280',
#   'AvailableDSP': '220',
#   'AvailableFF': '106400',
#   'AvailableLUT': '53200',
#   'AvailableURAM': '0'},
#  'TimingReport': {'WNS': -2.156,
#   'TNS': -5730.399,
#   'WHS': 0.006,
#   'THS': 0.0,
#   'WPWS': 3.75,
#   'TPWS': 0.0}}

# ON HARDWARE
# Classified 1500 samples in 0.571867 seconds (2622.9875128307804 inferences / s)
# Latency - 0.571867

# Latency - 3.5910563468933105
# Throughput - 417.7043897675618 images / sec

# Accuracy QKeras, CPU:     0.8946666666666667
# Accuracy hls4ml, pynq-z2: 0.892

def model3():

    input_shape = (32,32,3)
    x = x_in = Input(shape=input_shape)

###### 0
    x = QConv2DBatchnorm(
            8,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_quantizer="quantized_bits(4,0,alpha=1)",
            bias_quantizer="quantized_bits(4,0,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,
            name='convbn_{}'.format(0),
        )(x)

    x = QActivation('quantized_relu(4)', name='relu_conv_%i' % 0)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(0))(x)

###### 1
    x = QConv2DBatchnorm(
        8,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(4,0,alpha=1)",
        bias_quantizer="quantized_bits(4,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(1),
    )(x)

    x = QActivation('quantized_relu(4)', name='relu_conv_%i' % 1)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(1))(x)

######## 2
    x = QConv2DBatchnorm(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(4,0,alpha=1)",
        bias_quantizer="quantized_bits(4,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(2),
    )(x)

    x = QActivation('quantized_relu(4)', name='relu_conv_%i' % 2)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(2))(x)


    x = Flatten()(x)

########## 0
    x = QDense(
        42,
        kernel_quantizer="quantized_bits(4,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % 0,
        use_bias=False,
    )(x)

    x = BatchNormalization(name='bn_dense_{}'.format(0))(x)

    x = QActivation('quantized_relu(4)', name='relu_dense_%i' % 0)(x)

########## 1
    x = QDense(
        64,
        kernel_quantizer="quantized_bits(4,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % 1,
        use_bias=False,
    )(x)

    x = BatchNormalization(name='bn_dense_{}'.format(1))(x)

    x = QActivation('quantized_relu(4)', name='relu_dense_%i' % 1)(x)

    x = Dense(int(43), name='output_dense')(x)
    x_out = Activation('softmax', name='output_softmax')(x)
    qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
    return qmodel


# reduce filters and 4 bit activation
# {'CSynthesisReport': {'TargetClockPeriod': '10.00',
#   'EstimatedClockPeriod': '8.750',
#   'BestLatency': '24421',
#   'WorstLatency': '24554',
#   'IntervalMin': '14338',
#   'IntervalMax': '24526',
#   'BRAM_18K': '60',
#   'DSP': '75',
#   'FF': '42170',
#   'LUT': '85446',
#   'URAM': '0',
#   'AvailableBRAM_18K': '280',
#   'AvailableDSP': '220',
#   'AvailableFF': '106400',
#   'AvailableLUT': '53200',
#   'AvailableURAM': '0'},
#  'TimingReport': {'WNS': -2.041,
#   'TNS': -3399.926,
#   'WHS': 0.049,
#   'THS': 0.0,
#   'WPWS': 3.75,
#   'TPWS': 0.0}}

# ON HARDWARE
# Classified 1500 samples in 0.358902 seconds (4179.41387899761 inferences / s)
# Latency - 0.358902

# Latency - 3.2930285930633545
# Throughput - 455.50773630077055 images / sec

# Accuracy QKeras, CPU:     0.826
# Accuracy hls4ml, pynq-z2: 0.8313333333333334
def model4():
    input_shape = (32,32,3)
    x = x_in = Input(shape=input_shape)

###### 0
    x = QConv2DBatchnorm(
            4,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_quantizer="quantized_bits(6,0,alpha=1)",
            bias_quantizer="quantized_bits(6,0,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,
            name='convbn_{}'.format(0),
        )(x)

    x = QActivation('quantized_relu(6)', name='relu_conv_%i' % 0)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(0))(x)

###### 1
    x = QConv2DBatchnorm(
        4,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(1),
    )(x)

    x = QActivation('quantized_relu(6)', name='relu_conv_%i' % 1)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(1))(x)

######## 2
    x = QConv2DBatchnorm(
        8,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(2),
    )(x)

    x = QActivation('quantized_relu(6)', name='relu_conv_%i' % 2)(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(2))(x)


    x = Flatten()(x)

########## 0
    x = QDense(
        42,
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % 0,
        use_bias=False,
    )(x)

    x = BatchNormalization(name='bn_dense_{}'.format(0))(x)

    x = QActivation('quantized_relu(6)', name='relu_dense_%i' % 0)(x)

########## 1
    x = QDense(
        64,
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % 1,
        use_bias=False,
    )(x)

    x = BatchNormalization(name='bn_dense_{}'.format(1))(x)

    x = QActivation('quantized_relu(6)', name='relu_dense_%i' % 1)(x)

    x = Dense(int(43), name='output_dense')(x)
    x_out = Activation('softmax', name='output_softmax')(x)
    qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
    return qmodel



# {'CSynthesisReport': {'TargetClockPeriod': '10.00',
#   'EstimatedClockPeriod': '8.750',
#   'BestLatency': '110489',
#   'WorstLatency': '110637',
#   'IntervalMin': '34817',
#   'IntervalMax': '110593',
#   'BRAM_18K': '48',
#   'DSP': '1',
#   'FF': '26685',
#   'LUT': '65831',
#   'URAM': '0',
#   'AvailableBRAM_18K': '280',
#   'AvailableDSP': '220',
#   'AvailableFF': '106400',
#   'AvailableLUT': '53200',
#   'AvailableURAM': '0'},
#  'TimingReport': {'WNS': -0.915,
#   'TNS': -434.001,
#   'WHS': 0.018,
#   'THS': 0.0,
#   'WPWS': 3.75,
#   'TPWS': 0.0}}

# ON HARDWARE
# Classified 1500 samples in 1.5736080000000001 seconds (953.2234203181478 inferences / s)
# Latency - 1.5736080000000001

# Latency - 4.11785101890564
# Throughput - 364.2676709558667 images / sec

# Accuracy QKeras, CPU:     0.852
# Accuracy hls4ml, pynq-z2: 0.8546666666666667

def model5():
    input_shape = (32,32,3)
    x = x_in = Input(shape=input_shape)


    x = QConv2DBatchnorm(
        int(4),
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(6,2,alpha=1)",
        bias_quantizer="quantized_bits(6,2,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(0),
    )(x)

    #x = ZeroPadding2D(padding=(1,1), name='zero_pad_{}'.format(0))(x)
    x = QActivation('quantized_relu(6)', name='relu_conv_0')(x)

    x = QConv2DBatchnorm(
        int(4),
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(6,2,alpha=1)",
        bias_quantizer="quantized_bits(6,2,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(1),
    )(x)

    #x = ZeroPadding2D(padding=(2,2), name='zero_pad_{}'.format(1))(x)
    x = QActivation('quantized_relu(6)', name='relu_conv_1')(x)

    x = QConv2DBatchnorm(
        int(4),
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(6,2,alpha=1)",
        bias_quantizer="quantized_bits(6,2,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(2),
    )(x)

    #x = ZeroPadding2D(padding=(0,0), name='zero_pad_{}'.format(2))(x)
    x = QActivation('quantized_relu(6)', name='relu_conv_2')(x)

    x = QConv2DBatchnorm(
        int(4),
        kernel_size=(3, 3),
        strides=(4, 4),
        kernel_quantizer="quantized_bits(6,2,alpha=1)",
        bias_quantizer="quantized_bits(6,2,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(3),
    )(x)

    #x = ZeroPadding2D(padding=(1,1), name='zero_pad_{}'.format(3))(x)
    x = QActivation('quantized_relu(6)', name='relu_conv_3')(x)

    x = QConv2DBatchnorm(
        int(4),
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_quantizer="quantized_bits(6,2,alpha=1)",
        bias_quantizer="quantized_bits(6,2,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='convbn_{}'.format(4),
    )(x)
    x = QActivation('quantized_relu(6)', name='relu_conv_4')(x)

    x = Flatten()(x)
    x = QDense(
            43,
            kernel_quantizer="quantized_bits(6,2,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            name='output_dense',
            use_bias=False,
        )(x)

    x_out = Activation('softmax', name='output_softmax')(x)
    qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
    return qmodel