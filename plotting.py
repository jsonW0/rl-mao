import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from mao_env.mao import *

def plot_means(rewards_df,title,save_name=None,show=True):
    fig, ax = plt.subplots()
    ax.bar(np.arange(rewards_df.shape[1]), rewards_df.mean(), yerr=rewards_df.sem(), capsize=8, alpha=0.8)
    ax.set_ylabel(f'Mean Score, n={rewards_df.shape[0]}')
    ax.set_xticks(np.arange(len(rewards_df.mean())))
    ax.set_xticklabels(rewards_df.columns)
    plt.xticks(rotation=30)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y",which="major")
    ax.grid(axis="y",which="minor",alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    if show:
        plt.show()

def plot_wins(rewards_df,title,save_name=None,show=True):
    fig, ax = plt.subplots()
    win_counts = Counter(np.argmax(rewards_df.to_numpy(), axis=1))
    ax.bar(np.arange(rewards_df.shape[1]), [win_counts[i] for i in np.arange(rewards_df.shape[1])], capsize=8, alpha=0.8)
    ax.set_ylabel(f'Number of Wins, n={rewards_df.shape[0]}')
    ax.set_xticks(np.arange(len(rewards_df.mean())))
    ax.set_xticklabels(rewards_df.columns)
    plt.xticks(rotation=30)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y",which="major")
    ax.grid(axis="y",which="minor",alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    if show:
        plt.show()

def compute_scopes(results,config):
    game = MaoGame(config)
    scopes = {i:defaultdict(lambda: {"passed": 0, "failed": 0}) for i in range(config.num_players)}
    for episode in results:
        for player_id in range(config.num_players):
            states = episode["episode_states"][player_id]
            for state in states:
                hand = state["hand"]
                last_card = state["obs"]
                if len(last_card)>0: # Ignoring first card since that is always legal!
                    game.played_cards = last_card
                    for potential_card in hand:
                        for rule in config.validity_rules:
                            if rule(game,potential_card):
                                scopes[player_id][rule.__name__]["passed"]+=1
                            else:
                                scopes[player_id][rule.__name__]["failed"]+=1
    return {player_id:{rule:value["failed"]/(value["passed"]+value["failed"]) for rule,value in rules.items()} for player_id,rules in scopes.items()}

def compute_confidence(results,config):
    game = MaoGame(config)
    scopes = {i:defaultdict(lambda: {"passed": 0, "failed": 0}) for i in range(config.num_players)}
    # for episode in results:
    for player_id in range(config.num_players):
        states = results[player_id]
        # print("LENGTH",len(states))
        for state in states:
            last_card = state["obs"]
            action = state["action"]
            if len(last_card)>0: # Ignoring first card since that is always legal!
                game.played_cards = last_card
                for rule in config.validity_rules:
                    if rule(game,action):
                        scopes[player_id][rule.__name__]["passed"]+=1
                    else:
                        scopes[player_id][rule.__name__]["failed"]+=1
    return {player_id:{rule:value["failed"]/(value["passed"]+value["failed"]) for rule,value in rules.items()} for player_id,rules in scopes.items()}
