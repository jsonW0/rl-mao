import argparse
import os
import time
import dill as pickle
import pandas as pd
from mao_env.env import MaoEnv
from mao_env.mao import *
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.data import Collector, Batch
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy
from manual_policy import UnoPolicy
from tqdm import tqdm
from plotting import *

'''
Usage:
python eval_agent.py --save_name dqn --agents Random Random Random DQN,agents/uno/dqn --num_evals 1
python eval_agent.py --save_name all_manual --agents Manual Manual Manual Manual

'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=False, help="Save results to filename")
    parser.add_argument("--agents", type=str, required=False, nargs='+', help="Specify agents. Either name or a name,path or name,folder if is a trained policy. Only one folder may be specified.")
    parser.add_argument("--num_evals", type=int, default=1, help="Number of evaluations to do. >1 => specified a folder")
    parser.add_argument("--config", type=str, default="uno", help="Select config to use for evaluation")
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--save_game_text", action='store_true', help="Save game text")
    parser.add_argument("--render", type=float, default=1e-9, help="Render speed (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Set random seed")
    args = parser.parse_args()

    results = []

    for trial in tqdm(range(args.num_evals)):

        # Load agents
        names = []
        agents = []
        for agent in args.agents:
            if agent=="Random":
                names.append("Random")
                agents.append(RandomPolicy())
            elif agent=="Manual":
                names.append("Manual")
                agents.append(UnoPolicy())
            elif len(agent.split(","))==2:
                names.append(agent.split(",")[0])
                if ".pickle" in agent.split(",")[1]:
                    with open(agent.split(",")[1],"rb") as f:
                        agents.append(pickle.load(f))
                else:
                    with open(agent.split(",")[1]+f"/{agent.split(',')[1].split('/')[-1]}_agent_{trial}.pickle","rb") as f:
                        agents.append(pickle.load(f))
            else:
                raise NotImplementedError(f"{agent} not recognized and/or path not given")

        # Run environment
        os.makedirs(f"results/{args.config}/{args.save_name}", exist_ok=True)
        with open(f"results/{args.config}/{args.save_name}/game.txt","w") as f:
            pass
        config = Config(4,["Alpha","Beta","Gamma","Delta"],52)
        if args.save_game_text:
            env = MaoEnv(config, render_mode="file", save_render=f"results/{args.config}/{args.save_name}/game.txt")
        else:
            env = MaoEnv(config)

        env = PettingZooEnv(env)
        policy = MultiAgentPolicyManager(agents, env)
        env = DummyVectorEnv([lambda: env])
        collector = Collector(policy, env)
        result = collector.collect(n_episode=args.n_episodes, render=args.render)
        results.append(result)

    # Save results
    with open(f"results/{args.config}/{args.save_name}/results.pickle","wb") as f:
        pickle.dump(results,f)
    df = pd.DataFrame(np.concatenate([result['rews'] for result in results],axis=0),columns=names)
    df.to_csv(f"results/{args.config}/{args.save_name}/scores.csv",index=False)
    plot_means(df,title=f"{args.save_name},{args.config},N={args.n_episodes*args.num_evals}",save_name=f"results/{args.config}/{args.save_name}/mean_score.png",show=False)
    plot_wins(df,title=f"{args.save_name},{args.config},N={args.n_episodes*args.num_evals}",save_name=f"results/{args.config}/{args.save_name}/win_count.png",show=False)

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Time Elapsed: {end-start:.2f} seconds")
