# Reinforcement Learning Agents on Rule Discovery
Pavan Pandurangi, William Shi, Jason Wang

## Project Description

How quickly are standard reinforcement learning algorithms able to learn
newly added rules to the game of Mao? How does the complexity of the base game and the complexity
of the added rule impact the rate of rule discovery?

## Setup

Clone the repository.

```shell
conda create --name rl-mao
conda activate rl-mao
conda install python=3.9
pip install pettingzoo
pip install tianshou
pip install dill
```

## Project Organization

The `mao_env` folder contains the Mao game environment. We allow for rules that can be classified as either "validity" rule or "dynamics" rule. A validity rule is a Python function that determines whether a card can be played. A dynamics rule is a Python function that is called after a card is successfully played, and may alter game state.

`manual_policy.py` contains `UnoPolicy`, which is a policy that simply plays Uno as choosing uniformly any card that matches the suit or number.

`eval_agent.py` is our evaluation utility, and `train_agent.py` is a training utility.