


class NeuralPlayerDummy():
	def __init__(self, config = None, env = None):
		self.config = config.config_NeuralPlayer
		self.env = env
		self.agent =  None
		self.preprocessor = None
		self._init_agent(self.config.config_Agent)
		self._init_preprocessor(self.config.config_Preprocessing)



	def _init_preprocessor(self, config_Preprocessing = None):
		self.preprocessor = None


	def _init_agent(self, agentConfig = None):
		self.agent = None


	def _train_agent(self):
		self.agent.train()


	def _is_over_race(self, info):
		return False


	def do_races(self, episodes = None):
		for e in range(1, episodes):
			state, _, _, _ = self.env.reset()
			processed_state = self.preprocessor.preprocess(state)

			end_race = False
			while (not end_race):
				action = self.agent.get_action(processed_state, e)
				# steering, throttle = action[0], action[1]
				new_state, reward, done, info = self.env.step(action)
				new_processed_state = self.preprocessor.process(new_state)
				# self.agent.memory.add(blabla)
				processed_state = new_processed_state
				end_race  = self._is_over_race(info) or done
			if (e % self.config.train_frequency == 0):
				self.train_agent()
