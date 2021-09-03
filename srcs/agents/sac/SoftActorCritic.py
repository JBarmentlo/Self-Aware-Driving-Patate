import torch
import Policy
import Qfunction


class SoftActorCritic():
	"""
	Inspiration from: https://spinningup.openai.com/en/latest/algorithms/sac.html
	"""
	def __init__(self):
		self.learning_rate = learning_rate

		self.policy = Policy()

		self.phi_1 = Qfunction()
		self.phi_2 = Qfunction()

		self.discount_factor = 0.9

	def get_action(self, s_t):
		gaussians = self.policy.model(s_t)
		throttle, steering = self.policy.draw_actions(*gaussians)
		return throttle, steering

	def phi_min(self, state, action):
		phi_1 = self.phi_1.model(state, action) 
		phi_2 = self.phi_2.model(state, action)
		phi = torch.min(phi_1, phi_2)
		return phi

	def compute_targets(self, r, s_t1, done):
		gaussians = self.policy.model(s_t1)
		action = self.policy(*gaussians)
		probability = self.policy.policy_probability(gaussians, action)

		Qvalue = self.phi_min(s_t1, action)

		df = self.discount_factor
		lr = self.learning_rate

		targets = r + df * (1 - done) * (Qvalue - lr * torch.log(probability))

		return targets

	def policy_update(self, state_t):

		gaussians = self.policy.model(state_t)
		action_t = self.policy(*gaussians)
		probability = self.policy.policy_probability(gaussians, action_t)

		Qvalue = self.phi_min(state_t, action_t)

		lr = self.learning_rate

		loss = (Qvalue - lr * torch.log(probability))




	def train(self, replay_bufer):
		if len(replay_bufer) < self.batch_size:
			return
		for i in range(len(replay_bufer) // self.batch_size):
			# * Create batch
			batch = []
			for _ in range(self.batch_size):
				elem = replay_bufer.pop()
				# print(elem)
				batch.append(elem)
			
			state_t, action_t, reward_t, state_t1, done, _ = zip(*batch)

			# line 12
			targets = self.compute_targets(reward_t, state_t1)

			# line 13
			self.phi_1.train(state_t, action_t, targets)
			self.phi_2.train(state_t, action_t, targets)

			# line 14
			self.policy.train(state_t, self.phi_min)

			# line 15
			self.phi_1.soft_update()
			self.phi_2.soft_update()

		replay_bufer.clear()


if __name__ == "__main__":
	SoftActorCritic()
	