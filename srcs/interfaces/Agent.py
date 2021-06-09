params = {"lol" : "ye"}

# TODO : Define params template


class IAgent():
    def __init__(self, params = params):
        self.params = params


    def choose_action(self, state):
        '''
            Choose an action based on a state

            Args:
                state ([ndarray]): 

            Output:
                action ([steering, throttle])
        '''
        raise NotImplementedError
    