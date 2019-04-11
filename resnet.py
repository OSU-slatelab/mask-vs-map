import tensorflow as tf
import numpy as np
import tfsignal

def filterbank(inputs):
    inputs = tf.squeeze(inputs, axis=0)
    inputs = tf.nn.relu(inputs)+1e-6

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
    log_mel = tf.expand_dims(tf.log(mel_spectrograms + 1e-6), axis=0)

    return log_mel


class ResNet():
    """
    ResNet-style architecture for speech denoising.
    """

    def __init__(self,
        inputs,
        output_dim, 
        output_type = [],
        filters     = [128, 128, 256, 256],
        fc_layers   = 2,
        fc_nodes    = 2048,
        activation  = tf.nn.relu,
        dropout     = 0.3,
        log_mel     = False,
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

        # Store hyperparameters
        self.inputs = inputs
        self.activation = activation
        self.dropout = dropout
        self.training = tf.placeholder(tf.bool)
        self.fc_nodes = fc_nodes
        self.fc_layers = fc_layers

        batch_size = tf.shape(inputs)[0]
        channels = inputs.get_shape().as_list()[1]
        length = tf.shape(inputs)[2]
        features = inputs.get_shape().as_list()[3]

        if log_mel:
            block = filterbank(inputs)
        else:
            block = inputs

        # Convolutional part
        for i, f in enumerate(filters):
            with tf.variable_scope("block{0}".format(i)):
                block = self.conv_block(block, f)

        # Change to [batch, length, filters, feats]
        flat = tf.transpose(block, [0, 2, 1, 3])

        # Smush last two dimensions
        shape = flat.get_shape().as_list()
        flat = tf.reshape(flat, [batch_size, length, shape[2] * shape[3] // 16])
        flat = tf.concat((flat, self.inputs[:,0]), axis = -1)
        flat = tf.layers.dropout(flat, rate=dropout, training=self.training)

        # Fully conntected part
        out_shape = [batch_size, channels, length, output_dim]
        self.outputs = self.fully_connected(flat, out_shape)

        # Compute outputs
        input_min = tf.reduce_min(self.inputs)
        self.scale_vars = []
        if 'masking' in output_type:
            self.masking = tf.identity(self.outputs)
            self.outputs = tf.multiply(tf.sigmoid(self.outputs), self.inputs - input_min) + input_min
            if 'fidelity' in output_type:
                self.outputs = self.scale(self.outputs, name = 'mask', scale_init = 0.5)
                self.fidelity = self.fully_connected(flat, out_shape, name="fc_out")
                self.outputs += self.scale(self.fidelity, name = 'map', scale_init = 0.5)
            else:
                self.outputs = self.scale(self.outputs, name = 'mask')
        elif 'fidelity' in output_type:
            self.fidelity = tf.identity(self.outputs)
            if 'map-as-mask-mimic' in output_type:
                masked_by_map = tf.multiply(self.maskify(self.outputs), self.inputs - input_min) + input_min
                self.outputs = self.scale(masked_by_map, 'mask')

    def fully_connected(self, inputs, output_shape, name="fc"):
        fc = tf.identity(inputs)
        for i in range(self.fc_layers):
            with tf.variable_scope(name + str(i)):
                fc = tf.layers.dense(fc, self.fc_nodes, self.activation)
                fc = tf.layers.dropout(fc, rate = self.dropout, training = self.training)

        outputs = tf.layers.dense(fc, output_shape[-1])
        outputs = tf.reshape(outputs, shape = output_shape)

        return outputs

    def scale(self, inputs, name, scale_init = 1.0, shift_init = 0.0):
        scale = tf.get_variable(name + "_scale", (), initializer=tf.constant_initializer(scale_init))
        shift = tf.get_variable(name + "_shift", (), initializer=tf.constant_initializer(shift_init))
        self.scale_vars += [scale, shift]
        return inputs * scale + shift

    def maskify(self, inputs):
        in_min = tf.reduce_min(inputs)
        in_max = tf.reduce_max(inputs) - in_min
        return (self.outputs - in_min) / in_max

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
            strides     = 2 if downsample else 1,
            #dilation_rate = 2 if downsample else 1,
            padding     = 'same',
            data_format = 'channels_first',
            activation  = self.activation if not downsample else None,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
        )

        #layer = tf.contrib.layers.instance_norm(
        #    inputs        = layer,
        #    activation_fn = self.activation if not downsample else None,
        #    data_format   = 'NCHW',
        #)

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


