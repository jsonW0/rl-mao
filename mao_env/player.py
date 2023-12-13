from collections import defaultdict
class Player:
    def __init__(self,name):
        '''
        Class to hold player-specific information such as points and hand
        :param name: string representing player name
        '''
        self.name = name
        self.points = 0
        self.rules_violated = defaultdict(int)
        self.hand = []
    def __repr__(self):
        return f"{self.name}({self.hand},{self.points})"