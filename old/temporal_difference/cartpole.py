import argparse
import gymnasium as gym
from agents import CartPoleQLearning, CartPoleSARSA, CartPoleExpectedSARSA, CartPoleDoubleQLearning
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--n_bins', type=int, default=10, help="Number of bins in the discretisation of the state space.")
    argparser.add_argument('--strategy', type=str, default='uniform', help="Binnning strategy.")
    argparser.add_argument('--n_pretrain', type=int, default=10000, help="Number of episodes in the pretraining phase.")
    argparser.add_argument('--algorithm', type=str, default='q_learning', help="String in {'q_learning', 'sarsa', 'expected_sarsa, double_q_learning'}, the algorithm to use.")
    argparser.add_argument('--alpha', type=float, default=0.1, help="Step size parameter for Q-Learning.")
    argparser.add_argument('--epsilon_start', type=float, default=1, help="Initial value of epsilon.")
    argparser.add_argument('--epsilon_stop', type=float, default=0.1, help="Final value of epsilon.")
    argparser.add_argument('--decay_rate', type=float, default=1e-4, help="Decay rate of epsilon.")
    argparser.add_argument('--n_train', type=int, default=100000, help="Number of training episodes.")
    argparser.add_argument('--reward_stop', type=float, default=400, help="Minimum mean reward to stop training.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--save_gif', type=int, default=0, help="If 1, save gif of the tested agent, 0 otherwise.")
    
    args = argparser.parse_args()

    assert args.algorithm in set(['q_learning', 'sarsa', 'expected_sarsa', 'double_q_learning']), "algorithm should be in {'q_learning', 'sarsa', 'expected_sarsa', 'double_q_learning'}."

    # Train.
    env = gym.make('CartPole-v1')
    if args.algorithm == 'q_learning':
        agent = CartPoleQLearning(env=env, gamma=args.gamma, n_bins=args.n_bins, strategy=args.strategy)

    elif args.algorithm == 'sarsa':
        agent = CartPoleSARSA(env=env, gamma=args.gamma, n_bins=args.n_bins, strategy=args.strategy)

    elif args.algorithm == 'expected_sarsa':
        agent = CartPoleExpectedSARSA(env=env, gamma=args.gamma, n_bins=args.n_bins, strategy=args.strategy)

    elif args.algorithm == 'double_q_learning':
        agent = CartPoleDoubleQLearning(env=env, gamma=args.gamma, n_bins=args.n_bins, strategy=args.strategy)
    
    agent.pretrain(n_pretrain=args.n_pretrain)
    start_time = time()
    agent.train(alpha=args.alpha, epsilon_start=args.epsilon_start, epsilon_stop=args.epsilon_stop, decay_rate=args.decay_rate, 
                n_train=args.n_train, print_iter=100, reward_stop=args.reward_stop)
    print('Execution time :', time() - start_time)
    
    # Test.
    env = gym.make('CartPole-v1', render_mode='human')
    agent.test(env, n_episodes=args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif.
    if args.save_gif:
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        agent.save_gif(env, file_name='../gifs/cartpole.gif', n_episodes=3, duration=30)
        env.close()