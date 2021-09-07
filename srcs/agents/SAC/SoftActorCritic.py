import torch
from .Policy import Policy
from .Qfunction import Qfunction


class SoftActorCritic():
	"""
	Inspiration from: https://spinningup.openai.com/en/latest/algorithms/sac.html
	"""
	def __init__(self):
		self.learning_rate = 1e-3

		self.policy = Policy()

		self.phi_1 = Qfunction()
		self.phi_2 = Qfunction()

		self.discount_factor = 0.9
		self.batch_size = 32

	def get_action(self, s_t):
		gaussians = self.policy.model(s_t)
		# print(f"{gaussians = }")
		throttle, steering = self.policy.draw_actions(*gaussians)
		actions = [	float(throttle.cpu().detach()),
					float(steering.cpu().detach())]
		return actions

	def phi_min(self, state, action):
		phi_1 = self.phi_1.model(state, action) 
		phi_2 = self.phi_2.model(state, action)
		phi = torch.min(phi_1, phi_2)
		return phi

	def compute_targets(self, s_t1, r, done):
		gaussians = self.policy.model(s_t1)
		action = self.policy.draw_actions(*gaussians)
		probability = self.policy.policy_probability(gaussians, action)

		action = torch.cat((action[0], action[1]), dim=1)
		Qvalue = self.phi_min(s_t1, action)

		df = self.discount_factor
		lr = self.learning_rate

		targets = r + df * (1 - done) * (Qvalue - lr * torch.log(probability))

		return targets

	def policy_update(self, state_t):

		gaussians = self.policy.model(state_t)
		action_t = self.policy(*gaussians)
		probability = self.policy.policy_probability(gaussians, action_t)

		action_t = torch.cat((action_t[0], action_t[1]), dim=1)
		Qvalue = self.phi_min(state_t, action_t)

		lr = self.learning_rate

		loss = (Qvalue - lr * torch.log(probability))


	def train(self, dataset):
		print(f"{len(dataset) = }")
		if len(dataset) < self.batch_size:
			return False
		mini_batchs = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
		i = 0 
		torch.autograd.set_detect_anomaly(True)
		for state_t, action_t, state_t1, reward_t, done in mini_batchs:
			print(f"STARTING BATCH {i}")
			# print(f"{state_t.shape = }")
			# print(f"{state_t1.shape = }")
			# print(f"{action_t.shape = }")
			# line 12
			# targets_1 = self.compute_targets(state_t1.clone().detach(), reward_t.clone().detach(), done.clone().detach()).type(torch.float32)
			# targets_2 = self.compute_targets(state_t1.clone().detach(), reward_t.clone().detach(), done.clone().detach()).type(torch.float32)
			targets = self.compute_targets(state_t1, reward_t, done).type(torch.float32)

			# print(f"{state_t.dtype}")
			# print(f"{action_t.dtype}")
			# print(f"{targets_1.dtype}")
			targets_1 = targets.clone().detach()
			targets_2 = targets.clone().detach()
			state_t_1 = state_t.clone().detach()
			state_t_2 = state_t.clone().detach()
			action_t_1 = action_t.clone().detach()
			action_t_2 = action_t.clone().detach()
			# line 13
			self.phi_1.train(state_t_1, action_t_1, targets_1)
			self.phi_2.train(state_t_2, action_t_2, targets_2)

			# line 14
			self.policy.train(state_t, self.phi_min)

			# line 15
			self.phi_1.soft_update()
			self.phi_2.soft_update()

			i += 1

		return True


if __name__ == "__main__":
	SoftActorCritic()
