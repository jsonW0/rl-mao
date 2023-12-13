import argparse
import os
import time
import numpy as np
import pandas as pd
import random
import datetime
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy, DQNPolicy, PPOPolicy, A2CPolicy, PGPolicy, NPGPolicy, TRPOPolicy, C51Policy, RainbowPolicy, BasePolicy
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger, BaseLogger
from tianshou.utils.net.common import Net, ActorCritic
from torch.utils.tensorboard import SummaryWriter
from mao_env.env import MaoEnv
from mao_env.mao import *
from manual_policy import ManualPolicy
from detailed_collector import DetailedCollector
from utils import *
import dill as pickle
import string
from plotting import *
from log_trainer import logoffpolicy_trainer

'''
Usage:
python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules uno
'''

def set_seed(seed, train_envs, test_envs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_envs.seed(seed)
    test_envs.seed(seed)

def get_env(args,render_mode=None):
    config = Config(4,string.ascii_letters[:len(args.opponents)+1],52,[rule_names_to_functions[rule] for rule in args.rules])
    env = MaoEnv(config, render_mode=render_mode)
    env = PettingZooEnv(env)
    return env

def get_agents(args, trial, agent_learn=None, optim=None):
    env = get_env(args)
    config = Config(4,string.ascii_letters[:len(args.opponents)+1],52,[rule_names_to_functions[rule] for rule in args.rules])
    observation_space = (env.observation_space["observation"])
    action_space = env.action_space

    if agent_learn is None:
        def dist(p, m):
            return MaskedCategorical(logits=p, mask=m)
        if args.resume_path is None:
            net = Net(observation_space.shape, action_space.n, hidden_sizes=args.hidden_sizes, device="cpu").to("cpu")
            if args.policy=="DQN":
                optim = torch.optim.Adam(net.parameters(), lr=args.lr)
                agent_learn = DQNPolicy(net, optim, args.gamma, estimation_step=args.n_step, target_update_freq=args.target_update_freq)
            elif args.policy=="C51":
                optim = torch.optim.Adam(net.parameters(), lr=args.lr)
                agent_learn = C51Policy(net, optim, args.gamma, 52, estimation_step=args.n_step, target_update_freq=args.target_update_freq)
            elif args.policy=="Rainbow":
                optim = torch.optim.Adam(net.parameters(), lr=args.lr)
                agent_learn = RainbowPolicy(net, optim, args.gamma, 52, estimation_step=args.n_step, target_update_freq=args.target_update_freq)
            elif args.policy=="PG":
                actor = Actor(net, action_space.n)
                optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
                agent_learn = PGPolicy(actor,optim,dist)
            elif args.policy=="A2C":
                actor = Actor(net, action_space.n)
                critic = Critic(net)
                optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr, eps=1e-5)
                agent_learn = A2CPolicy(actor, critic, optim, dist)
            elif args.policy=="NPG":
                actor = Actor(net, action_space.n)
                critic = Critic(net)
                optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr, eps=1e-5)
                agent_learn = NPGPolicy(actor, critic, optim, dist)
            elif args.policy=="TRPO":
                actor = Actor(net, action_space.n)
                critic = Critic(net)
                optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr, eps=1e-5)
                agent_learn = TRPOPolicy(actor, critic, optim, dist)
            elif args.policy=="PPO":
                actor = Actor(net,action_space.n)
                critic = Critic(net)
                optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr, eps=1e-5)
                agent_learn = PPOPolicy(actor,critic,optim,dist)
            else:
                raise NotImplementedError(f"Policy '{args.policy}' not recognized")
        else:
            with open(f"{args.resume_path}/agent_{trial}.pickle", 'rb') as f:
                agent_learn = pickle.load(f)
            optim = torch.optim.Adam(agent_learn.model.parameters(), lr=args.lr, eps=1e-5)
    # if args.resume_path:
        # agent_learn.load_state_dict(torch.load(f"{args.resume_path}/agent_{trial}.pickle"))


    agents = [agent_learn]
    names = ["Self"]
    for agent in args.opponents:
        if agent == "Random":
            names.append("Random")
            agents.append(RandomPolicy())
        elif agent == "Manual":
            names.append("Manual")
            agents.append(ManualPolicy(config))
        elif agent == "Self":
            names.append("Self")
            agents.append(agent_learn)
        elif len(agent.split(",")) == 2:
            names.append(agent.split(",")[0])
            with open(agent.split(",")[1], "rb") as f:
                agents.append(pickle.load(f))
        else:
            raise NotImplementedError(f"{agent} not recognized and/or path not given")

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents, config


def train_agent(args, trial, agent_learn=None):
    # Setup environments
    train_envs = DummyVectorEnv([(lambda: get_env(args)) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([(lambda: get_env(args)) for _ in range(args.test_num)])

    # Set seed
    set_seed(args.seed, train_envs, test_envs)

    # Generate agents
    policy, optim, agents, config = get_agents(args, trial, agent_learn=agent_learn)

    # Collect
    train_collector = DetailedCollector(policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)), exploration_noise=True)
    test_collector = DetailedCollector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # Logger

    logger = WandbLogger(config=args,project="rl-mao")
    writer = SummaryWriter(os.path.join(args.logdir, "wandb"))
    writer.add_text("args", str(args))
    logger.load(writer)
    # log_path = os.path.join(args.logdir, "tf")
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(args))
    # logger = TensorboardLogger(writer)

    # Callbacks
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            now = datetime.datetime.now()
            date_path = "{0}-{1}-{2}-{3}-{4}-{5}{6}".format(now.year, now.month, now.day, now.hour, now.minute, now.second, ".pth")
            model_save_path = os.path.join(args.logdir, date_path)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, 0]

    # train
    if args.policy in {"DQN","C51","Rainbow"}:
        result = logoffpolicy_trainer(
            config=config,
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=10*args.test_num,
            batch_size=args.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            # stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            update_per_step=args.update_per_step,
            logger=logger,
            test_in_train=False,
            reward_metric=reward_metric,
        )
    elif args.policy in {"PG","A2C","PPO","NPG","TRPO"}:
        result = onpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            # train_fn=train_fn,
            # test_fn=test_fn,
            # stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            update_per_step=args.update_per_step,
            logger=logger,
            test_in_train=False,
            reward_metric=reward_metric,
        )
    return result, policy.policies[agents[0]]

def main():
    parser = argparse.ArgumentParser()

    # Setup arguments
    parser.add_argument("--policy", type=str, required=True, help="Name of policy to train")
    parser.add_argument("--opponents", type=str, required=True, nargs='+', help="Specify opponents. Either name or a name,path if is a trained policy")
    parser.add_argument("--save_name", type=str, required=True, help="Save results to filename")
    parser.add_argument("--num_agents", type=int, default=1, help="Number of agents to train")
    parser.add_argument("--render", type=float, default=1e-9, help="Render speed (seconds)")
    parser.add_argument("--logdir", type=str, default="log/dqn", help="Directory to log training info to")
    parser.add_argument("--rules", type=str, default=[], nargs="*", help="Select rules to use for training")
    parser.add_argument("--resume_path", type=str, required=False, help="To start from an existing model, pass a model directory here")

    # Model-specific parameters
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--n_step", type=int, default=3, help="Number of steps to lookahead before updating q-function")
    parser.add_argument("--target_update_freq", type=int, default=320, help="Number of steps between every update of the parameters of the target network")
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[512, 512, 512], help="DQN network architecture")

    # Training parameters
    parser.add_argument("--epoch", type=int, default=3, help="Number of epochs for training, unless stop function is set")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    #TODO: why do we have multiple environments?
    parser.add_argument("--training_num", type=int, default=10, help="Number of training environments")
    parser.add_argument("--test_num", type=int, default=10, help="Number of testing environments")
    parser.add_argument("--eps_train", type=float, default=0.1, help="During training, exploration rate in epsilon-greedy algorithm")
    parser.add_argument("--eps_test", type=float, default=0.05, help="During testing, exploration rate in epsilon-greedy algorithm")
    parser.add_argument("--buffer_size", type=int, default=20000, help="Size of replay buffer")
    parser.add_argument("--step_per_epoch", type=int, default=1000, help="The number of transitions collected per epoch")
    parser.add_argument("--step_per_collect", type=int, default=10, help="The number of transitions the collector would collect before the network update")
    parser.add_argument("--update_per_step", type=float, default=0.1, help="Number of times the policy network would be updated per transition after (step_per_collect) transitions are collected")
    parser.add_argument("--repeat_per_collect", type=int, default=4, help="The number of repeat time for policy learning, for example, set it to 2 means the policy needs to learn each given batch data twice.")
    parser.add_argument("--seed", type=int, default=42, help="Set random seed")

    args = parser.parse_args()

    results, trained_agents = [],[]
    for trial in range(args.num_agents):
        print(f"Training Agent {trial+1}/{args.num_agents}")
        result, trained_agent = train_agent(args,trial)
        results.append(result)
        trained_agents.append(trained_agent)

    # Save Model and Training Args
    os.makedirs(f"agents/rules={','.join(sorted(args.rules))}/{args.save_name}", exist_ok=True)
    for trial in range(args.num_agents):
        with open(f"agents/rules={','.join(sorted(args.rules))}/{args.save_name}/agent_{trial}.pickle","wb") as f:
            pickle.dump(trained_agents[trial],f)
    with open(f"agents/rules={','.join(sorted(args.rules))}/{args.save_name}/args.pickle","wb") as f:
        pickle.dump(args,f)

if __name__ == "__main__":
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()
    end = time.perf_counter()
    print(f"Time Elapsed: {end-start:.2f} seconds")