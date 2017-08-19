import trafficSystem

class Game:

    def __init__(self):
        trafficSystem.init(True)

    def step(self,action):
        return trafficSystem.actionStepAndGetState(action)

    def print_stats(self):
        trafficSystem.print_stats()