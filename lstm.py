import tensorflow as tf
import numpy as np
import tfsignal

def filterbank(inputs):
    inputs = tf.squeeze(inputs, axis=0)
    #inputs = tf.nn.relu(inputs)+1e-6
    inputs = tf.exp(inputs)

    # Convert to log-mel
    num_bins = inputs.shape[-1].value
    sample_rate, lower_edge_hertz, upper_edge_hertz, num_mel_bins = 16000.0, 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tfsignal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
    )
    mel_spectrograms = tf.tensordot(inputs, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(inputs.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel = tf.expand_dims(tf.log(mel_spectrograms + 0.1), axis=0)

    return log_mel


class BiLSTM():
    """
    ResNet-style architecture for speech denoising.
    """

    def __init__(self,
        inputs,
        output_shape, 
        layers     = 4,
        units      = 320,
        activation = tf.nn.relu,
        dropout    = 0.3,
    ):
        """
        Build the graph.

        Parameters
        ----------
        inputs : Placeholder
            Spectral inputs to this model, of the shape (batchsize, frames, frequencies)
        output_shape : int
            Size of the output
        filters : list of ints
            Size of each block
        fc_layers : int
            Number of fully-connected hidden layers
        fc_nodes : int
            Number of units to put in each fc hidden layer
        activation : function
            Function to apply before conv layers as an activation
        dropout : float
            Fraction of filters and nodes to drop

        Returns
        -------
        Tensor
            Outputs of the dropnet model
        """
        projection = 256

        # Store hyperparameters
        self.inputs = inputs
        self.activation = activation
        self.dropout = dropout
        self.training = tf.placeholder(tf.bool)

        block = filterbank(inputs)
        
        for i, f in enumerate([128]):
            with tf.variable_scope("block{}".format(i)):
                block = self.conv_block(block, f)

        # LSTM is length x batch x feats
        lstm_inputs = tf.transpose(block, [2, 0, 1, 3])
        shape = lstm_inputs.get_shape().as_list()
        lstm_inputs = tf.reshape(lstm_inputs, [-1, shape[1], shape[2] * shape[3]]) 
        self.seq_len = tf.shape(lstm_inputs)[0]

        # Project down
        lstm_inputs = tf.layers.dense(lstm_inputs, projection)

        model = tf.contrib.cudnn_rnn.CudnnLSTM(layers, units, direction='bidirectional', dropout=dropout)
        #input_h = tf.zeros([layers*2, 1, units], dtype=tf.float32)
        #input_c = tf.zeros([layers*2, 1, units], dtype=tf.float32)

        #params = tf.Variable(tf.random_uniform([model.params_size()], -0.1, 0.1), validate_shape=False)

        #lstm_outputs, out_h, out_c = model(lstm_inputs, input_h=input_h, input_c=input_c, params=params, is_training=True)
        lstm_outputs, _ = model(lstm_inputs, training = True)

        fc = tf.layers.dropout(lstm_outputs, training = self.training)

        # Project down
        fc = tf.layers.dense(fc, projection, activation = tf.nn.relu)
        fc = tf.layers.dropout(fc, training = self.training)

        #for i in range(fc_layers):
        #    fc = tf.layers.dense(fc, fc_units, name = f"fc{i}")
        #    fc = tf.layers.dropout(fc, training = self.training)
        
        self.outputs = tf.layers.dense(fc, -np.prod(output_shape))
        self.outputs = tf.reshape(self.outputs, list(output_shape))

    def conv_layer(self, inputs, filters, downsample=False, dropout=True):
        """
        One convolutional layer, with convolutional dropout.

        Parameters
        ----------
        inputs : Tensor
            input to convolutional layer
        filters : int
            size of convolutional layer
        downsample : boolean
            Whether or not this is a downsampling layer

        Returns
        -------
        Tensor
            outputs of convolutional layer
        """

        # Apply convolution
        layer = tf.layers.conv2d(
            inputs      = inputs,
            filters     = filters,
            kernel_size = 3,
            strides     = 3 if downsample else 1,
            #dilation_rate = 2 if downsample else 1,
            padding     = 'same',
            data_format = 'channels_first',
            activation  = self.activation if not downsample else None,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
        )

        # Dropout
        if dropout:
            shape = tf.shape(layer)
            layer = tf.layers.dropout(
                inputs      = layer,
                rate        = self.dropout,
                training    = self.training,
                noise_shape = [shape[0], filters, 1, 1],
            )

        return layer


    def conv_block(self, inputs, filters):
        """
        A residual block of three conv layers.

        First layer down-samples using a stride of 2. Output of the third
        layer is added to the first layer as a residual connection.

        Parameters
        ----------
        inputs : Tensor
            Input to this conv block
        filters : int
            Width of this conv block

        Returns
        -------
        Tensor
            Output of this conv block
        """
        
        # Down-sample layer
        with tf.variable_scope("downsample"):
            downsampled = self.conv_layer(inputs, filters, downsample=True)

        # 1st conv layer
        with tf.variable_scope("conv1"):
            conv1 = self.conv_layer(self.activation(downsampled), filters)

        # 2nd conv layer
        with tf.variable_scope("conv2"):
            conv2 = self.conv_layer(conv1, filters, dropout=False)

        return downsampled + conv2

