import functools
import logging
import numpy as np
import gymnasium
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Sequence, Dict, MultiBinary

from mao import *

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
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
        self.agent_selection = None
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
        return Dict({
            "observation": Dict({
                "hand": MultiBinary([52]),
                "hand_lengths": Box(low=0,high=np.PINF,shape=self.config.num_players),
                "played_cards": MultiBinary([52]),
                "points": Box(low=np.NINF,high=np.PINF,shape=self.config.num_players),
            }),
            "action_mask": MultiBinary([52]),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(52) #52 kinds of cards to play

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
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

    def reset(self, seed=None, return_info=True, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        """
        self.game = MaoGame(self.config)
        self.game.deal()
        self.agents = self.possible_agents[:]
        observations = {agent: self.game.get_observations(self.agent_name_mapping[agent]) for agent in self.agents}

        if not return_info:
            return observations
        else:
            infos = {agent: {} for agent in self.agents}
            return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # process card plays
        actions_list = [0 for _ in range(self.config.num_players)]
        for agent, action in actions.items():
            actions_list[self.agent_name_mapping[agent]] = action
        self.game.play(actions_list)

        # termination and truncation (https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

        terminations = {agent: False for agent in self.agents}

        env_truncation = False  # no truncation since games take constant number of turns
        truncations = {agent: env_truncation for agent in self.agents}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        # rewards for all agents are placed in the rewards dictionary to be returned
        points = self.game.get_rewards()
        rewards = {agent: points[i] for i,agent in enumerate(self.agents)}

        if self.config.num_players <= 3:
            if self.game.card_num == 10:
                if self.game.round_num == 3:
                    terminations = {agent: True for agent in self.agents}
                else:
                    self.game.deal()

        # query for next observation
        observations = {agent: self.game.get_observations(self.agent_name_mapping[agent]) for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode is not None:
            self.render()

        for agent in terminations:
            if terminations[agent]:
                self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos

if __name__ == "__main__":
    # AEC API Test
    mao_env = env(Config(3,["Alpha","Beta","Gamma"],52))
    api_test(mao_env)

    # AEC API Random Sampler
    mao_env.reset()
    for agent in mao_env.agent_iter():
        observation, reward, termination, truncation, info = mao_env.last()
        print(agent)
        print(mao_env.unwrapped.game.decode_observation(observation["observation"]))
        print()
        if termination or truncation:
            action = None
        else:
            action = mao_env.action_space(agent).sample(observation["action_mask"])  # this is where you would insert your policy
    
        mao_env.step(action)
    mao_env.close()