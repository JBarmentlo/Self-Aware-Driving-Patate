# Needed for training the network
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers


class GaussianPolicy():
	"""
		Inspiration from: https://towardsdatascience.com/a-minimal-working-example-for-continuous-policy-gradients-in-tensorflow-2-0-d3413ec38c6b
		 Tutorial written the 18th of Aug 2020
	"""

	def __init__(self, input_shape=(1,), bias_mu=0.0, bias_sigma=0.55, learning_rate=0.001):
		"""
			Summary:
				In the continuous variant, we usually draw actions from a Gaussian distribution;
				The goal is to learn an appropriate mean μ and a standard deviation σ

				Once obtaining this output (μ and σ), an action a is randomly drawn from the corresponding Gaussian distribution.
				Thus, we have a=μ(s)+σ(s)ξ , where ξ ∼ 𝒩(0,1).

			Args:
				input_shape ([type]):
					The input is the state s 
						or
					a feature array ϕ(s), followed by one or more hidden layers that transform the input
				output_size ([type]):
					The output being μ and σ
		"""
		# Create actor network
		# bias 0.0 yields mu=0.0 with linear activation function
		self.bias_mu = 0.0
		self.bias_mu = bias_mu
		# bias 0.55 yields sigma=1.0 with softplus activation function
		self.bias_sigma = 0.55
		self.bias_sigma = bias_sigma

		self.lr = learning_rate

		self.actor_network = self.build_model(
                   				input_shape,  # input dimension is (1,) for testor
               					self.bias_mu,
               					self.bias_sigma)
		self.opt = keras.optimizers.Adam(learning_rate=self.lr)

		# For debugging purposes
		self.loss_ = 0
		self.mu_ = 0
		self.sigma_ = 0

	def build_model(self, input_shape, bias_mu, bias_sigma):
		"""
			Construct the actor network with mu and sigma as output
		"""
		inputs = layers.Input(shape=input_shape)

		hidden1 = layers.Dense(5,
                         activation="relu",
                         kernel_initializer=initializers.he_normal())(inputs)

		hidden2 = layers.Dense(5,
                         activation="relu",
                         kernel_initializer=initializers.he_normal())(hidden1)

		mu = layers.Dense(1,
                    activation="linear",
                    kernel_initializer=initializers.Zeros(),
                    bias_initializer=initializers.Constant(bias_mu))(hidden2)

		sigma = layers.Dense(1,
                       activation="softplus",
                       kernel_initializer=initializers.Zeros(),
                       bias_initializer=initializers.Constant(bias_sigma))(hidden2)

		actor_network = keras.Model(inputs=inputs, outputs=[mu, sigma])

		return actor_network

	def choose_action(self, state):
		# Obtain mu and sigma from network
		self.mu_, self.sigma_ = self.actor_network(state)

		# Draw action from normal distribution
		action = tf.random.normal([1], mean=self.mu_, stddev=self.sigma_)

		return action


	def custom_loss_gaussian(self, state, action, reward):
		"""[summary]
			Weighted Gaussian log likelihood loss function for RL

		Args:
			state ([type]): [description]
			action ([type]): [description]
			reward ([type]): [description]

		Returns:
			[type]: loss_actor
			Next step will be to apply it
		"""
		# * Predict mu and sigma with actor network
		mu, sigma = self.actor_network(state)

		# * Compute Gaussian pdf value
		pdf_value = tf.exp(-0.5 * ((action - mu) / (sigma))**2) * \
				1 / (sigma * tf.sqrt(2 * np.pi))

		# * Convert pdf value to log probability
		log_probability = tf.math.log(pdf_value + 1e-5)

		# * Compute weighted loss
		self.loss_ = - reward * log_probability

		return self.loss_


	def update(self, state, action, reward):
		# * MATHEMATICS:
			# *
			# * After taking our action a, we observe a corresponding reward signal v.
			# * Together with some learning rate α, we may update the weights into a direction that improves the expected reward of our policy
			# *
			# * Update rule:
			# * DELTA_mu_th(s) = alpha * v * (a - mu_th(s)) / std_th**2
			# * DELTA_std_th(s) = alpha * v * ((a - mu_th(s))**2 - std_th**2) / std_th**3
			# *
			# * Linear approx:
			# * mu_th(s) = th ** (T_phi(s))
			# *
			# * For NeuralNets: not as starightforward
			# * -> Trained by minimizing loss function
			# * 	-> is computed by MSE
			# *
			# * For the policy, we need to define a 'pseudo loss function' to update net
			# *
			# * Update rule in its generic form:
			# *		DELTA_th = alpha * NABLA_th (log(pi_th(a | s)) * v)
			# *
			# *		Update form gradient decent, hence minus sign
			# *			Loss(a, s, v) = -log(pi_th(a | s)) * v
			# *
			# *			Intution: log yields neg nb if input < 1
			# *				If action w/ small prob && high reward:
			# *					WE WANT large loss
			# *
			# *		Here, we can substitute pi_th by Gaussian Probability density function:
			# *
			# *			GPD() = ( 1 / (std (2 pi)**(1/2)) ) * e^( -1/2 * ((a - mu) / std)**2 )
			# *			Loss(a, s, v) = -log(GPD()) * v
			# *

		# * IMPLEMTATION:
			# *
			# *		Unfortunately, it's not directly applicable in TS 2.0
			# *
			# *			Normally ->
			# *						model.compile(loss='mse',optimizer=opt)
			# *						model.fit || model.train_on_batch
			# *
			# *			But ->
			# *					- Gaussian log likelihood loss function is not a default one in TensorFlow 2.0 (but it's in Theano)
			# *						-> Have to create a custom one
			# *
			# *					- TS requires 2 args: y_true, y_predicted
			# *						But -> we have 3 (we need to multiply w/ reward)
			# *
			# *			SEE: implementation in custom_loss_gaussian(state, action, reward)
			# *
			# *		How to apply our loss function update ?
			# *
			# *			GradientTape -> novel addition to TF 2.0
			# *				What is this ?
			# *					It records the forward steps on a 'tape'
			# *						-> so it can apply automatic differentiation
			# *
			# *				Updating approach:
			# *
			# *					1st :
			# *						Forward pass (actor net) with custom loss func
			# *							values are memorized
			# *						Calculate loss
			# *
			# *					2nd :
			# *						with .trainable_variables() -> we recall the weights of forward pass
			# *						tape.gradient calculates all the gradients
			# *							it plugs in loss value && trainable variables
			# *
			# *					3rd :
			# *						with optimizer.apply_gradients() -> update net weights
			# *							different optimizers are possible: SGD, Adam, RMSprop,...
			# *
			# *

		"""
			Compute and apply gradients to update network weights
		"""
		with tf.GradientTape() as tape:
			# Compute Gaussian loss with custom loss function
			loss_value = self.custom_loss_gaussian(state,
												action,
												reward)

			# Compute gradients for actor network
			grads = tape.gradient(loss_value,
								self.actor_network.trainable_variables)

			# Apply gradients to update network weights
			self.opt.apply_gradients(zip(grads,
										self.actor_network.trainable_variables))
