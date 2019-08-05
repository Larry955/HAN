from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K
"""
Implementation of Attention
"""

class AttLayer(Layer):
	def __init__(self, attention_dim, **kwargs):
		self.init = initializer.get('normal')
		self.supports_masking = True
		self.attention_dim = attention_dim
		super(AttLayer, self).__init__()

	def build(self, input_shape):
		assert len(input_shape) == 3
		self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
		self.b = K.variable(self.init((self.attention_dim, )))
		self.u = K.variable(self.init((self.attention_dim, 1)))
		self.trainable_weights = [self.W, self.b, self.u]
		super(AttLayer, self).build(input_shape)

	def compute_mask(self, inputs, mask=None):
		return mask
	
	def call(self, x, mask=None):
		# size of x: [batch_size, sel_len, attention_dim]
		# size of u: [batch_size, attention_dim]
		# uit = tanh(xW + b)
		uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
		ait = K.dot(uit, self.u)
		ait = K.squeeze(ait, -1)

		ait = K.exp(ait)

		if mask is not None:
			# Cast the mask to floatX to avoid float64 upcasting in theano
			ait *= K.cast(mask, K.floatx())
		ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		ait = K.expand_dims(ait)
		weighted_input = x * ait
		output = K.sum(weighted_input, axis=1)

		return output

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])

	def get_config(self):
		config = {
			'attenion_dim': self.attention_dim
		}
		base_config = super(AttLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


	

