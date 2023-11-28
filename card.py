class Card:
    def __init__(self,number,suit):
        '''
        Card class
        :param number: number
        :param suit: suit
        '''
        self.number = number
        self.suit = suit
        self.id = (self.number-1) + 13*suit_to_num[self.suit]
    def __repr__(self):
        return f"{number_to_string[self.number]}{suit_to_string[self.suit]}"
    # Functions such that cards can be hashable objects
    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        return self.__hash__()==other.__hash__()
    
number_to_string = {
    1: "A",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
}
suit_to_string = {
    "C": "♣",
    "D": "♦",
    "H": "♥",
    "S": "♠" 
}
suit_to_num = {
    "C": 0,
    "D": 1,
    "H": 2,
    "S": 3
}

def parse_human_input(card):
    return Card(int(card[:-1]),card[-1]).id