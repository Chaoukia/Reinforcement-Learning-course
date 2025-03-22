from q_learning import *
from time import time
import gym
import argparse

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--n_bins', type=int, default=10, help="Number of bins used to discretise the state space.")
    argparser.add_argument('--n_initialise', type=int, default=10000, help="Number of iterations in the pretraining phase used to define the bins.")
    argparser.add_argument('--n_train', type=int, default=20000, help="Number of training episodes.")
    argparser.add_argument('--epsilon_start', type=float, default=1, help="Initial value of epsilon.")
    argparser.add_argument('--epsilon_stop', type=float, default=0.1, help="Final value of epsilon.")
    argparser.add_argument('--decay_rate', type=float, default=2e-6, help="Decay rate of epsilon.")
    argparser.add_argument('--log_dir', type=str, default='runs_qlearning', help="Directory in which to store the tensorboard events.")
    argparser.add_argument('--thresh', type=float, default=450, help="Average return over which we terminate training.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--file_save', type=str, default='q_learning.pkl', help="Name of the pickle file where to save the q-values.")
    argparser.add_argument('--save_gif', type=int, default=0, help="If 1, save gif of the tested agent, 0 otherwise.")
    
    args = argparser.parse_args()
    
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = CartPole(state_dim=state_dim, n_actions=n_actions, gamma=args.gamma)
    agent.initialize(env, n_bins=args.n_bins, n_episodes=args.n_initialise)

    # Training.
    start_time = time()
    agent.train(env, n_episodes=args.n_train, epsilon_start=args.epsilon_start, epsilon_stop=args.epsilon_stop, decay_rate=args.decay_rate, log_dir=args.log_dir, thresh=args.thresh)
    print('Training time :', time() - start_time)
        
    # Test.
    env = gym.make("CartPole-v1", render_mode='human')
    agent.test(env, n_episodes=args.n_test)
    env.close()
    
    # Save gif.
    if args.save_gif:
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        agent.save_gif(env, file_name='../gifs/cartpole.gif')
        env.close()