from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
# from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.initializers import identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
# from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from collections import deque
import numpy as np
import random
from utils import linear_unbin
from copy import deepcopy


def build_model_ValueNetwork(input_shape, output_size, learning_rate):
	"""
	This model will be an approximator of the Value Function to estimate the Expected Return of an episode from a state
	"""
	model = Sequential()
	model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",
			   input_shape=input_shape))  # 80*80*4
	model.add(Activation('relu'))
	model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	# TODO: check activation layer for output
	model.add(Dense(output_size, activation="linear"))
	adam = Adam(lr=learning_rate)
	# TODO: check which loss to choose
	model.compile(loss='mse', optimizer=adam)
	return model


def custom_loss_gaussian(state, action, reward):
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
	mu, sigma = actor_network(state)

	# * Compute Gaussian pdf value
	pdf_value = tf.exp(-0.5 * ((action - mu) / (sigma))**2) * \
            1 / (sigma * tf.sqrt(2 * np.pi))

	# * Convert pdf value to log probability
	log_probability = tf.math.log(pdf_value + 1e-5)

	# * Compute weighted loss
	loss_actor = - reward * log_probability

	return loss_actor

class GaussianPolicy():
	"""
	Inspiration from: https://towardsdatascience.com/a-minimal-working-example-for-continuous-policy-gradients-in-tensorflow-2-0-d3413ec38c6b
	"""
	def __init__(self, input_shape, output_size):
		"""[summary]
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
		pass

	def update(self):
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
		# *
		pass

class SoftActorCritic():
	"""
	Inspiration from: https://spinningup.openai.com/en/latest/algorithms/sac.html
	"""
	def __init__(self, input_shape, output_size, learning_rate):
		# Q functions estimators:
		self.phi_1 = build_model_ValueNetwork(input_shape, output_size, learning_rate)
		self.phi_2 = build_model_ValueNetwork(input_shape, output_size, learning_rate)

		self.discount_factor = 0.9
		self.lr_qfunc = 1e-4
		pass

	def policy_predict(self, s_t):
		pass

	def policy_choose(self, s_t):
		pred = self.policy_predict(s_t)
		a_t = pred[random.randint(0, len(pred - 1))]
		return a_t

	def qfunc_predict(self, s_t1, a_t1, which=0):
			# Implementation is not clear if we need to sample a_t1 twice
			if which == 1:
				q_values = self.phi_1.predict(s_t1, a_t1)
			elif which == 2:
				q_values = self.phi_2.predict(s_t1, a_t1)
			else:
				q_values_1 = self.phi_1.predict(s_t1, a_t1)
				q_values_2 = self.phi_2.predict(s_t1, a_t1)
				q_values = np.min(q_values_1, q_values_2)
			return q_values

	def qfuncs_update(self, state_t, targets):
		# * The update should be on new networks to allow for soft update
		#	TODO: Update should be done with MSE
		phi_1 = deepcopy(self.phi_1)
		phi_2 = deepcopy(self.phi_2)

		phi_1.train_on_batch(state_t, targets)
		phi_2.train_on_batch(state_t, targets)

		return phi_i, phi_2


    def soft_net_update(self, net_old, net_new, TAU=0.8):
		# TODO: put TAU in config.py
        ''' Update the targer gradually. '''
        # Extract parameters  
        net_old_params = net_old.named_parameters()
    	net_new_params = net_new.named_parameters()
        
        dic_net_new_params = dict(net_new_params)

        for old_name, old_param in net_old_params:
            if old_name in old_param:
                dic_net_new_params[old_name].data.copy_(
                    (TAU)*predi_param.data + (1-TAU)*old_param[predi_param].data)

        net_new.load_state_dict(net_old.state_dict())
		return net_new

	def compute_targets(self, r, s_t1, done):
		a_t1 = self.policy_predict(s_t1)
		# If done, eon is not necessary
		# 	If none might even break code
		eon = self.qfunc_predict(s_t1, a_t1, which=0) - \
						(self.lr_qfunc * np.ln(self.policy_predict(a_t1, s_t1)))
		targets = r + self.discount_factor  (1 - done) * eon
		return targets

	def train(self, replay_bufer, batch_size):
		for i in range(len(replay_bufer) // batch_size):
			# * Create batch
			batch = create_batch(replay_bufer, batch_size)
			state_t, action_t, reward_t, state_t1, done = batch

			# * Compute targets
			targets = self.compute_targets(reward_t, state_t1, done)

			# * Compute the update Q_functions estimators phi_1 & phi_2
			phi_1, phi_2 = self.qfuncs_update(state_t, targets)

			# * Update Policy, w/ gradient acent:
			# ? Not sure how to, reference back to pseudocode from here : https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode
			# TODO: Once policy is implemented
			a_t = self.policy(s_t)

			# * Soft update the target networks
			self.soft_net_update(self.phi_1.model, phi_1.model)
			self.soft_net_update(self.phi_2.model, phi_2.model)












# Open-AI: pseudocode ->
# alt="\begin{algorithm}[H]
#     \caption{Soft Actor-Critic}
#     \label{alg1}
# \begin{algorithmic}[1]
#     \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$
#     \STATE Set target parameters equal to main parameters $\phi_{\text{targ},1} \leftarrow \phi_1$, $\phi_{\text{targ},2} \leftarrow \phi_2$
#     \REPEAT
#         \STATE Observe state $s$ and select action $a \sim \pi_{\theta}(\cdot|s)$
#         \STATE Execute $a$ in the environment
#         \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
#         \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
#         \STATE If $s'$ is terminal, reset environment state.
#         \IF{it's time to update}
#             \FOR{$j$ in range(however many updates)}
#                 \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
#                 \STATE Compute targets for the Q functions:
#                 \begin{align*}
#                     y (r,s',d) &= r + \gamma (1-d) \left(\min_{i=1,2} Q_{\phi_{\text{targ}, i}} (s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s')\right), && \tilde{a}' \sim \pi_{\theta}(\cdot|s')
#                 \end{align*}
#                 \STATE Update Q-functions by one step of gradient descent using
#                 \begin{align*}
#                     & \nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi_i}(s,a) - y(r,s',d) \right)^2 && \text{for } i=1,2
#                 \end{align*}
#                 \STATE Update policy by one step of gradient ascent using
#                 \begin{equation*}
#                     \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B} \Big(\min_{i=1,2} Q_{\phi_i}(s, \tilde{a}_{\theta}(s)) - \alpha \log \pi_{\theta} \left(\left. \tilde{a}_{\theta}(s) \right| s\right) \Big),
#                 \end{equation*}
#                 where $\tilde{a}_{\theta}(s)$ is a sample from $\pi_{\theta}(\cdot|s)$ which is differentiable wrt $\theta$ via the reparametrization trick.
#                 \STATE Update target networks with
#                 \begin{align*}
#                     \phi_{\text{targ},i} &\leftarrow \rho \phi_{\text{targ}, i} + (1-\rho) \phi_i && \text{for } i=1,2
#                 \end{align*}
#             \ENDFOR
#         \ENDIF
#     \UNTIL{convergence}
# \end{algorithmic}
# \end{algorithm}"
