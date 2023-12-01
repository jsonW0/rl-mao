import random
from collections import Counter, OrderedDict
import numpy as np
import copy
from typing import NamedTuple, List
from player import *
from card import *

class Config(NamedTuple):
    num_players: int
    player_names: List[str]
    deck_size: int

def uno_rules(game,card):
    return len(game.played_cards)==0 or game.played_cards[-1].suit==card.suit or game.played_cards[-1].number==card.number

def king_skips(game,card):
    if card.number==13:
        game.turn+=1
        game.turn%=game.config.num_players

class MaoGame:
    '''
    An instance of a Mao Game, containing all game rules and states
    '''
    def __init__(self,config):
        '''
        :param config: a namedtuple containing game setup specifications. It must contain the following attributes:
            num_players: int
            player_names: str
        '''
        self.config = config
        self.players = [Player(name) for name in config.player_names]
        self.deck = []
        self.played_cards = []
        self.round_num = 0
        self.card_num = 0
        self.turn = 0
        self.is_done = False
        self.validity_rules = [uno_rules]
        self.dynamics_rules = [] #[king_skips]

    def __repr__(self):
        return f"Game({self.round_num}-{self.card_num},{self.players})"

    def pprint(self,autoprint=True):
        '''
        Pretty print function of the game state
        '''
        output = f"Round {self.round_num}-{self.card_num}:\n\t" + f"Played Cards: {self.played_cards}\n\t" + "\n\t".join(
                f"{player.name}: {player.points}"
                f"\n\t\tHand: {player.hand}"
                 for player in self.players)
        if autoprint:
            print(output)
        else:
            return output

    def render(self): #TODO graphical interface
        pass

    def deal(self):
        '''
        This function should be called at the start of every round.
        It deals the appropriate set of cards to each player's hand.
        '''
        self.deck = [Card(num,suit) for num in range(1,14) for suit in ["C","D","H","S"]]
        random.shuffle(self.deck)
        for i in range(len(self.players)):
            self.players[i].hand = [self.deck.pop() for _ in range(7)]

    def draw(self, playerIndex):
        '''
        This function draws a card for playerID
        '''
        if len(self.deck) == 0:
            self.deck = self.played_cards[:-1]
            if len(self.deck)==0:
                return
            random.shuffle(self.deck)
            self.played_cards = [self.played_cards[-1]]
        self.players[playerIndex].hand.append(self.deck.pop())

    def play(self,action): #TODO add special cards
        '''
        Plays the cards each player has selected
        :param action: the card id chosen by the player
        '''
        if self.is_done:
            print("Tried to play done game!")
            self.turn+=1
            self.turn%=self.config.num_players
            return

        current_player = self.players[self.turn]
        played_card_index = [i for i,card in enumerate(current_player.hand) if card.id==action][0]
        played_card = current_player.hand[played_card_index]

        # Apply any validity rules
        valid = True
        for rule in self.validity_rules:
            if rule(self,played_card):
                pass
            else:
                valid = False
                self.draw(self.turn)
                print(f"Penalty to {self.players[self.turn].name} for violating rule {rule}")
        if valid:
            self.played_cards.append(current_player.hand.pop(played_card_index))

            # Apply any dynamics rules
            for rule in self.dynamics_rules:
                rule(self,played_card)

        # Win condition
        winners = [player for player in self.players if len(player.hand)==0]
        if len(winners)>0:
            self.is_done = True
            print(f"The winner is {winners}.")

        # Increment game counter variables
        self.turn += 1
        self.turn %= self.config.num_players
        self.card_num += 1

    def encode(self,cards):
        '''
        Convert a list of cards to a one-hot encoded vector [Number of Egg Nigiri, Number of Salmon Nigiri, ...]
        :param cards: list of card objects
        :return: np array of card counts
        '''
        one_hot = [0 for _ in range(52)]
        for card in cards:
            one_hot[card.id]+=1
        return np.array(one_hot,dtype=np.int8)
    
    def decode(self,one_hot):
        '''
        Convert a one-hot encoded vector [Number of Egg Nigiri, Number of Salmon Nigiri, ...] to a list of cards
        :param one_hot: list of card counts
        :return: list of card objects
        '''
        cards = []
        for i,index in enumerate(one_hot):
            cards.extend([id_to_card(i) for _ in range(int(index))])
        return cards
    
    def flatten_observation(self,observation):
        '''
        Get a dict of hand, hand_lengths, played_cards, and points, and return flattened numpy array
        '''
        flattened = np.concatenate([
            observation["hand"],
            observation["hand_lengths"],
            observation["played_cards"],
            observation["points"]
        ])
        return flattened  
     
    def unflatten_observation(self,flattened):
        observation = {}
        start_index = 0
        print(self.decode(flattened[start_index:start_index+52]))
        observation["hands"] = self.decode(flattened[start_index:start_index+52])
        start_index = 52
        observation["hand_lengths"] = flattened[start_index:start_index+self.config.num_players]
        start_index = 52+self.config.num_players
        observation["played_cards"] = self.decode(flattened[start_index:start_index+52])
        start_index = 1-4+self.config.num_players
        observation["points"] = flattened[start_index:start_index+self.config.num_players]
        return observation

    def get_observation(self,playerIndex):
        '''
        Get all observations observable by the 'playerIndex'th player
        :param playerIndex: the index of the player whose observations to return
        :return: OrderedDict observation with the following fields (observation order starts with self, increases modularly)
            hand: known cards encoded for each player in observation order
            played_cards: played cards
            points: number of points for each player in observation order
            round: round number
            card_num: card number
            action_mask: boolean mask [has EggNigiri in hand, has SalmonNigiri in hand, ...]
        '''
        observation_order = [(playerIndex + i) % self.config.num_players for i in range(self.config.num_players)]
        return OrderedDict(sorted({
            "observation": self.flatten_observation({
                "hand": self.encode(self.players[playerIndex].hand),
                "hand_lengths": np.array([len(self.players[i].hand) for i in range(len(self.players))],dtype=np.float32),
                "played_cards": self.encode(self.played_cards),
                "points": np.array([self.players[i].points for i in observation_order],dtype=np.float32),
            }),
            "action_mask": np.array(self.encode(set(self.players[playerIndex].hand)),dtype=np.int8),
        }.items()))
    
    def get_reward(self, playerIndex):
        length_of_hand = len(self.players[playerIndex].hand)
        if length_of_hand==0:
            return 10000
        else:
            return -length_of_hand

if __name__ == "__main__":
    game = MaoGame(Config(3,["Alpha","Beta","Gamma"],52))
    game.deal()
    while not game.is_done:
        game.pprint()
        card = input()
        game.play(parse_human_input(card))
        # game.play(game.players[game.turn].hand[0].id)
    # game.play([player.hand[0].id for player in game.players])
    # game.get_observations(0)
    game.pprint()
