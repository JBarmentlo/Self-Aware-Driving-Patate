import logging

Logger = logging.getLogger("RewardOpti")
Logger.setLevel(logging.DEBUG)
stream = logging.StreamHandler()
Logger.addHandler(stream)

class RewardOpti:
    def __init__(self, config_NeuralPlayer):
        self.config = config_NeuralPlayer
    
    def reset(self):
        pass
    
    def new_race_init(self, episode):
        pass

    def close_to_center(self, cte):
        self.reward += (self.config.cte_limit - abs((cte + self.config.cte_offset))) * self.config.cte_coef
        Logger.debug(f"reward center: ({self.config.cte_limit} - {abs(cte + self.config.cte_offset)}) * coef = [{(self.config.cte_limit - abs((cte + self.config.cte_offset))) * self.config.cte_coef}]")

    def go_fast(self, speed):
        self.reward += speed * self.config.speed_coef
        Logger.debug(f"reward speed: {speed} * {self.config.speed_coef} = [{speed * self.config.speed_coef}]")
    
    def goes_backward(self, action):
        if action[0] < 0:
            return (True)
    
    def sticks_and_carrots(self, action, infos, done):
        ### TODO: other infos that could be use: infos["pos"], infos["gyro"], infos["lidar"], infos["car"]
        self.reward = 0
        Logger.debug(f"done: {done}, infos['hit'] = {infos['hit']}")
        if done or self.goes_backward(action): #or infos["hit"] != None #TODO maybe diferentiate the sticks
            self.reward = self.config.reward_stick
            Logger.debug(f"reward stick: {self.reward}")
            return (self.reward)
        self.close_to_center(infos["cte"])
        self.go_fast(infos["speed"])
        Logger.debug(f"**** total reward: [{self.reward}]")
        return (self.reward)


        

  
    
    
    