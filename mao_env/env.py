import functools
import logging
import numpy as np
import gymnasium
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Sequence, Dict, MultiBinary
# from gymnasium.wrappers import FlattenObservation

from .mao import *

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.test import api_test


def env(config,render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = MaoEnv(config,render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    # if render_mode == "ansi":
        # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    # Flatten Dict
    # env = FlattenObservation(env)
    return env

class MaoEnv(AECEnv):
    metadata = {"render_modes": ["ansi","file"], "name": "mao"}

    def __init__(self, config, render_mode=None, save_render=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.config = config
        self.game = MaoGame(self.config)
        self.possible_agents = self.config.player_names
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.save_render = save_render

        if save_render is not None:
            logging.basicConfig(filename=save_render, encoding='utf-8', level=logging.INFO, format='%(message)s')

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # return Dict({
        #     "observation": MultiDiscrete([11 for _ in range(37 * self.config.num_players)]+ #hands
        #                      [11 for _ in range(37 * self.config.num_players)]+ #played_cards
        #                      [16 for _ in range(37 * self.config.num_players)]+ #desserts
        #                      [300 for _ in range(self.config.num_players)]+ #points
        #                      [4]+ #round
        #                      [11] #card_num
        #                     ),
        #     "action_mask": MultiBinary([37]),
        # })
        # return Dict({
        #     "observation": Dict({
        #         "hand": MultiBinary([52]),
        #         "hand_lengths": Box(low=0,high=np.PINF,shape=[self.config.num_players]),
        #         "played_cards": MultiBinary([52]),
        #         "points": Box(low=np.NINF,high=np.PINF,shape=[self.config.num_players]),
        #     }),
        #     "action_mask": MultiBinary([52]),
        # })
        return Dict({
            "observation": Box(low=np.NINF,high=np.PINF,shape=[104+2*self.config.num_players]),
            "action_mask": MultiBinary([52]),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(52) #52 kinds of cards to play

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return self.observations[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "human":
            self.game.pprint()
        elif self.render_mode == "file":
            logging.info(self.game.pprint(autoprint=False))
        else:
            gymnasium.logger.warn(
                f"You are calling render method with unsupported render mode {self.render_mode}."
            )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        random.seed(seed)
        self.game = MaoGame(self.config)
        self.game.deal()

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: self.game.get_observation(i) for i,agent in enumerate(self.agents)}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.game.turn]

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        current_agent = self.agents[self.game.turn]
        current_agent_id = self.game.turn

        # run one step in game
        self.game.play(action)
        
        # update agent 
        self.rewards = {agent: 0 for agent in self.agents}
        self.rewards[current_agent] = self.game.get_reward(current_agent_id)
        self._cumulative_rewards[current_agent] = 0
        self.observations = {agent: self.game.get_observation(i) for i,agent in enumerate(self.agents)}
        if self.game.is_done:
            broken = False
            for agent in self.agents:
                if agent==current_agent:
                    break
                if not self.terminations[agent]:
                    broken = True
            if not broken:
                self.terminations[self.agents[self.game.turn]] = True
        self._accumulate_rewards()
        self.agent_selection = self.agents[self.game.turn]
        
        self.render()

if __name__ == "__main__":
    # AEC API Test
    mao_env = env(Config(4,["Alpha","Beta","Gamma","Delta"],52),render_mode="human")
    # print(mao_env.observation_space(0).sample())
    # api_test(mao_env)

    # AEC API Random Sampler
    mao_env.reset()
    while not mao_env.game.is_done:
        observation, reward, termination, truncation, info = mao_env.last()
        print(mao_env.game.players[mao_env.game.turn])
        print(mao_env.unwrapped.game.unflatten_observation(observation["observation"]))
        print()
        if termination or truncation:
            action = None
        else:
            action = mao_env.action_space(mao_env.game.players[mao_env.game.turn]).sample(observation["action_mask"])  # this is where you would insert your policy
    
        mao_env.step(action)
    mao_env.close()