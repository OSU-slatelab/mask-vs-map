import tensorflow as tf
import numpy as np
import tfsignal

class DNN():
    """
    ResNet-style architecture for speech denoising.
    """

    def __init__(self,
        inputs,
        output_dim, 
        layers     = 2,
        units      = 2048,
        context    = 5,
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

        # Store hyperparameters
        self.inputs = inputs
        self.activation = activation
        self.dropout = dropout
        self.training = tf.placeholder(tf.bool)

        # Pad inputs for contextual frames
        padding = [[0, 0], [0, 0] [context, context], [0, 0]]
        padded_inputs = tf.pad(inputs, padding, "REFLECT")

        #self.final_input = padded_inputs

        # We want to apply the DNN to overlapping regions of the input... so use CNN to implement the DNN!
        # Use filter size of frames x frequency x units
        fc = tf.layers.conv2d(
            inputs      = padded_inputs,
            filters     = units,
            kernel_size = [2*context+1, padded_inputs.get_shape()[-1]],
            activation  = activation,
            data_format = 'channels_first',
            name        = 'fc0',
        )

        for i in range(1, layers):
            fc = tf.layers.dense(
                inputs     = fc,
                units      = units,
                activation = activation,
                name       = f"fc{i}",
            )
            fc = tf.layers.dropout(
                inputs   = fc,
                rate     = dropout,
                training = self.training,
            )

        self.outputs = tf.layers.dense(fc, output_dim)
        self.outputs = tf.reshape(self.outputs, [1, 1, -1, output_dim])

        # Compute outputs
        input_min = tf.reduce_min(self.inputs)
        self.scale_vars = []
        if 'masking' in output_type:
            self.masking = tf.identity(self.outputs)
            self.outputs = tf.multiply(tf.sigmoid(self.outputs), self.inputs - input_min) + input_min
            #if 'fidelity' in output_type:
            #    self.outputs = self.scale(self.outputs, name = 'mask', scale_init = 0.5)
            #    self.fidelity = self.fully_connected(flat, out_shape, name="fc_out")
            #    self.outputs += self.scale(self.fidelity, name = 'map', scale_init = 0.5)
            #else:
            #    self.outputs = self.scale(self.outputs, name = 'mask')
        elif 'fidelity' in output_type:
            self.fidelity = tf.identity(self.outputs)
            #if 'map-as-mask-mimic' in output_type:
            #    masked_by_map = tf.multiply(self.maskify(self.outputs), self.inputs - input_min) + input_min
            #    self.outputs = self.scale(masked_by_map, 'mask')
 
