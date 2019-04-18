import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from tensorflow.keras.layers import Layer
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import standard_ops
from tensorflow.keras import backend as K

class PolymorphicDenseBase(Layer):
    def __init__(self,
                 units,
                 modes,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(PolymorphicDenseBase, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units)
        
        # How many differnt weights+biases sets this layer will contain
        # Normal Dense layer implicitly has exactly 1 mode.
        self.modes = int(modes)
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True

    def build(self, input_shape, key_size, last_dim):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `PolymorphicDenseBase` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `PolymorphicDenseBase` '
                             'should be defined. Found `None`.')

        # Scalar that defines distance sensitivity when key is compared with
        # keys map
        self.similarity_sensitivity = self.add_weight(
            'similarity_sensitivity',
            shape=[self.modes],
            initializer=initializers.Ones(),
            regularizer=self.kernel_regularizer,
            dtype=self.dtype,
            trainable=True)

        # Keys map that is compared with generated keys 
        self.keys_map = self.add_weight(
            'keys_map',
            shape=[self.modes, key_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        # Inputs processing weights map
        self.kernels = self.add_weight(
            'kernels',
            shape=[self.modes, last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        if self.use_bias:
            # Inputs processing biases map
            self.biases = self.add_weight(
                'biases',
                shape=[self.modes, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.biases = None
        super(PolymorphicDenseBase, self).build(input_shape)

    def call(self, key, input_):

        def similarity(keys, keys_table):
            a = tf.expand_dims(keys, -2)
            dist = tf.math.sqrt(
                tf.math.reduce_sum(tf.pow(a - keys_table, 2), axis=-1)
            )
            return self.similarity_sensitivity / ((dist + 1.))

        # Now we compare our key with keys tensor and get the list
        # of m similarities scalars where m is modes count
        raw_similarity = similarity(key, self.keys_map)
        raw_similarity = tf.nn.softmax(raw_similarity)

        # Adding dimensions for proper multiplication with kernels tensor
        key_similarity = raw_similarity
        key_similarity = tf.expand_dims(key_similarity, -1)
        key_similarity = tf.expand_dims(key_similarity, -1)
        
        # Generating tensor of M weights where m is modes count. 
        # Each weight is taken from kernels tensor and then multiplied by 
        # corresponding similarity.
        weighted_weights = key_similarity * self.kernels

        # Reducing weighted weights table into one weight (kernel)
        # that will be used as a normal Dense layer kernel.
        kernel = tf.math.reduce_mean(weighted_weights, axis=-3)

        outputs = tf.math.reduce_sum(tf.expand_dims(input_, -1) * kernel, axis=-2)

        if self.use_bias:
            # Similar excercise as was done for kernels tensor above,
            # but for bias this time.
            key_similarity = raw_similarity
            key_similarity = tf.expand_dims(key_similarity, -1)
            weighted_biases = key_similarity * self.biases
            bias = tf.math.reduce_mean(weighted_biases, axis=-2)
            
            outputs = outputs + bias

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'modes': self.modes,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(PolymorphicDenseBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PolymorphicDense(PolymorphicDenseBase):

    def __init__(self,
                 units,
                 modes,
                 key_size = None,
                 activation=None,
                 use_bias=True,
                 use_key_bias=None,
                 kernel_initializer='glorot_uniform',
                 key_kernel_initializer=None,
                 bias_initializer='zeros',
                 key_bias_initializer=None,
                 kernel_regularizer=None,
                 key_kernel_regularizer=None,
                 bias_regularizer=None,
                 key_bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 key_kernel_constraint=None,
                 bias_constraint=None,
                 key_bias_constraint=None,
                 **kwargs):
        super(PolymorphicDense, self).__init__(
            units=units,
            modes=modes,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, **kwargs)
 
        # If key size was not provided then specifying it as a logarithm 
        # of modes count. In this case number of parameters will be O(m)
        # where m is modes count.
        if key_size is None:
            key_size = int(1. + np.log(modes))
        self.key_size = int(key_size)
        
        # If use_key_bias was not specified then defaulting to use_bias
        if use_key_bias is None:
            use_key_bias = use_bias
        self.use_key_bias = use_key_bias
        
        # If initalizer, reguralizer, contstraint values were not
        # provided for key kernel and bias then defaulting to what 
        # was provided for normal kernel and bias tensors
        if key_kernel_initializer is None:
            key_kernel_initializer = kernel_initializer
            
        if key_bias_initializer is None:
            key_bias_initializer = bias_initializer
            
        if key_kernel_regularizer is None:
            key_kernel_regularizer = kernel_regularizer
            
        if key_bias_regularizer is None:
            key_bias_regularizer = bias_regularizer
            
        if key_kernel_constraint is None:
            key_kernel_constraint = kernel_constraint
            
        if key_bias_constraint is None:
            key_bias_constraint = bias_constraint
        
        self.key_kernel_initializer = initializers.get(key_kernel_initializer)
        self.key_bias_initializer = initializers.get(key_bias_initializer)
        self.key_kernel_regularizer = regularizers.get(key_kernel_regularizer)
        self.key_bias_regularizer = regularizers.get(key_bias_regularizer)
        self.key_kernel_constraint = constraints.get(key_kernel_constraint)
        self.key_bias_constraint = constraints.get(key_bias_constraint)

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `PolymorphicDense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `PolymorphicDense` '
                             'should be defined. Found `None`.')
        
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        
        # Key generation weights
        self.key_kernel = self.add_weight(
            'key_kernel',
            shape=[last_dim, self.key_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        # Key generation bias
        self.key_bias = self.add_weight(
            'key_bias',
            shape=[self.key_size],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)

        
        super(PolymorphicDense, self).build(input_shape, 
                                            self.key_size, 
                                            last_dim)
        
    def call(self, inputs):
        rank = common_shapes.rank(inputs)

        # This is same as linear output of a normal dense layer but
        # it's not used as layer output. Instead, this value (key) is
        # used to calculate the actual weights.
        key = standard_ops.tensordot(inputs, self.key_kernel, [[rank - 1], [0]])
        
        if self.use_key_bias:
            key = key + self.key_bias
        
        return super(PolymorphicDense, self).call(key, inputs)

    def get_config(self):
        config = {
            'key_size': self.key_size,
            'use_key_bias': self.use_key_bias,
            'key_kernel_initializer': initializers.serialize(self.key_kernel_initializer),
            'key_bias_initializer': initializers.serialize(self.key_bias_initializer),
            'key_kernel_regularizer': regularizers.serialize(self.key_kernel_regularizer),
            'key_bias_regularizer': regularizers.serialize(self.key_bias_regularizer),
            'key_kernel_constraint': constraints.serialize(self.key_kernel_constraint),
            'key_bias_constraint': constraints.serialize(self.key_bias_constraint)
        }
        base_config = super(PolymorphicDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))