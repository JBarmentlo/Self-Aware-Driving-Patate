import torch
from .Policy import Policy
from .Qfunction import Qfunction
from torch import optim


class SoftActorCritic():
	"""
	Inspiration from: https://spinningup.openai.com/en/latest/algorithms/sac.html
	"""
	def __init__(self, config):
		self.config = config
		self.lr = config.lr

		self.policy = Policy(config.policy)

		self.phi_1 = Qfunction(config.Qfunction)
		self.phi_2 = Qfunction(config.Qfunction)

		self.discount_factor = config.discount_factor
		self.batch_size = config.batch_size

		self.update_step = 0
		self.delay_step = config.train_delay_step

	def _scale_action(self, a_t):
		throttle = 1
		steering = 0

		a_t[...,steering] = torch.tanh   (a_t[...,steering])
		a_t[...,throttle] = torch.sigmoid(a_t[...,throttle])

		a_t[...,throttle] = torch.max(a_t[...,throttle], torch.ones(a_t.shape)[...,throttle] * .25)
		return a_t

	def _unscale_action(self, a_t):
		throttle = 1
		steering = 0

		a_t[...,steering] = torch.arctanh(a_t[...,steering])
		a_t[...,throttle] = torch.logit  (a_t[...,throttle])
		return a_t

	def get_action(self, s_t):
		_, _, action, _ = self.policy.model.sample(s_t)
		action = action.detach().cpu()
		action = self._scale_action(action)
		return action

	def _min_QValue(self, state, action):
		minQvalues = torch.min(
				self.phi_1.model(state, action),
				self.phi_2.model(state, action)
		)
		return minQvalues

	def _compute_targets(self, s_t1, r, done):
		_, _, action, log_pi = self.policy.model.sample(s_t1)

		Qvalues = self._min_QValue(s_t1, action)

		alpha = self.policy.alpha
		df = self.discount_factor

		next_Qvalues = (Qvalues - alpha * log_pi)
		next_Qvalues = (1 - done) * next_Qvalues
		targets = r + df * next_Qvalues
		
		Qvalues = targets.detach().type(torch.float32)

		return Qvalues


	def train(self, dataset):
		if len(dataset) < self.batch_size:
			return False
		print(f"Train {len(dataset)}")
		mini_batchs = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
		torch.autograd.set_detect_anomaly(True)
		for state_t, action_t, state_t1, reward_t, done in mini_batchs:
			#Reshape
			done = done.view((done.shape[0], -1))
			reward_t = reward_t.view((reward_t.shape[0], -1))
			action_t = self._unscale_action(action_t)
			# line 12
			expected_Qvalues = self._compute_targets(state_t1, reward_t, done)

			# line 13
			self.phi_1.train(state_t, action_t, expected_Qvalues.detach())
			self.phi_2.train(state_t, action_t, expected_Qvalues.detach())

			if self.update_step % self.delay_step == 0:
				# line 14
				self.policy.train(state_t, self._min_QValue)
				# line 15
				self.phi_1.soft_update()
				self.phi_2.soft_update()

			self.policy.update_temperature(state_t)

		self.update_step += 1

		return True


if __name__ == "__main__":
	SoftActorCritic()
