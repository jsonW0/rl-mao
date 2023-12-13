python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules uno

python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules alt_color
python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules alt_suit
python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules one_larger

python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules six_larger

python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules six_larger alt_color --resume_path agents/rules=six_larger/dqn
python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules six_larger alt_suit --resume_path agents/rules=six_larger/dqn
python train_agent.py --num_agents 1 --save_name dqn --policy DQN --opponents Manual Manual Manual --epoch 100 --rules six_larger one_larger --resume_path agents/rules=six_larger/dqn