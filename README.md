# Reinforcement-Learning-course
This repository contains an introductory course to Reinforcement Learning (RL) with hands-on classic examples of agents trained on gym environments. We start with Dynamic Programming algorithms: Value Iteration, Q-Iteration and Policy Iteration which we use to train an agent on the FrozenLake environment, then we we move on to Q-Learning and the cartpole environment; for Deep RL, we implement DQN and train an agent on the LunarLander environment. I will provide notes explaining the motivations, details, advantages and limitations of each method, along with documented python scripts, for Deep RL, I will be using pytorch. This is an ongoing project and I will include many more algorithms such as Reinforce and Actor-Critic variants...

## Dynamic Programming

Use file ```frozen_lake.py``` to train a Dynamic Programming agent on the FrozenLake environment, argument ```algorithm``` specifies which algorithm to use between ```value_iteration```, ```q_iteration``` and ```policy_oteration```. Example:
```
python frozen_lake.py --map_name 4x4 --gamma 0.9 --algorithm policy_iteration --epsilon 1e-12 --n_train 1000 --n_test 10
```

## Q-Learning

Use file ```cartpole.py``` to train a Q-Learning agent on the cartpole environment, arguments ```n_bins``` and ```n_initialise``` are very important as they initialise the bins that will be used to discretise the state space. Example:
```
python cartpole.py --gamma 0.9 --n_bins 10 --n_initialise 10000 --n_train 20000 --epsilon_start 1 --epsilon_stop 0.1 --decay_rate 2e-6 --log_dir runs_qlearning --thresh 450 --n_test 10
```

## DQN

Use file ```lunar_lander.py``` to train a DQN agent on the LunarLander environment. Example:
```
python lunar_lander.py --train 1 --gamma 0.99 --n_pretrain 64 --n_train 1000 --epsilon_start 1 --epsilon_stop 0.01 --decay_rate 2e-5 --n_learn 5 --batch_size 64 --lr 1e-3 --max_tau 25 --thresh 250 --n_test 10
```

