import argparse
import multiprocessing as mp
import torch
import gymnasium as gym
from reinforcement_learning_course.deep_rl.a2c.examples import agents
from reinforcement_learning_course.deep_rl.a2c.algorithms import train_a2c_worker, share_adam_optimizer
from time import time


def train_a2c_cartpole(worker_id,
                    policy_network,
                    value_network,
                    policy_optimizer,
                    value_optimizer,
                    gamma,
                    n_workers,
                    t_max,
                    n_episodes,
                    alpha_entropy,
                    thresh,
                    print_iter,
                    log_dir,
                    barrier,
                    lock,
                    ) -> None:
    """Worker function for A2C training on CartPole-v1.

    Creates a CartPole environment and A2C agent, then delegates to the shared
    train_a2c_worker routine. Intended to be spawned as a separate process.

    Args:
        worker_id: Integer identifier for this worker process.
        policy_network: Shared policy network across all workers.
        value_network: Shared value network across all workers.
        policy_optimizer: Shared optimizer for the policy network.
        value_optimizer: Shared optimizer for the value network.
        gamma: Discount factor.
        n_workers: Total number of parallel workers.
        t_max: Maximum number of steps between gradient updates.
        n_episodes: Maximum number of training episodes.
        alpha_entropy: Entropy regularization coefficient.
        thresh: Mean return threshold for early stopping.
        print_iter: Number of episodes between progress prints.
        log_dir: Directory for TensorBoard logs.
        barrier: Synchronization barrier shared across workers.
        lock: Mutex lock for safe gradient accumulation.
    """
    env = gym.make("CartPole-v1")
    agent = agents.CartPoleA2C(env, n_workers, gamma)
    train_a2c_worker(worker_id, 
                     policy_network,
                     value_network, 
                     policy_optimizer, 
                     value_optimizer, 
                     env,
                     agent,
                     t_max,
                     n_episodes,
                     alpha_entropy,
                     thresh,
                     print_iter,
                     log_dir,
                     barrier, 
                     lock, 
                     )


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')

    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--n_workers', type=int, default=6, help="Number of workers.")
    argparser.add_argument('--n_train', type=int, default=10000, help="Number of training episodes.")
    argparser.add_argument('--t_max', type=int, default=5, help="Number of steps between two consecutive gradient updates.")
    argparser.add_argument('--lr_policy', type=float, default=1e-4, help="Learning rate for the policy network.")
    argparser.add_argument('--lr_value', type=float, default=1e-4, help="Learning rate for the value network.")
    argparser.add_argument('--alpha_entropy', type=float, default=0.0, help="Penalty on the entropy loss term.")
    argparser.add_argument('--thresh', type=float, default=400, help="Minimum mean reward to stop training.")
    argparser.add_argument('--log_dir', type=str, default='runs', help="Path where to log the tensorboard runs.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--print_iter', type=int, default=100, help="If 1, save gif of the tested agent, 0 otherwise.")
    argparser.add_argument('--path_gif', type=str, help="Path where to save the test gif. If not provided, no gif save will be performed.")
    
    args = argparser.parse_args()

    barrier = mp.Barrier(args.n_workers)
    lock = mp.Lock()
    start_time = time()

    # Initilizing the shared parameters of the policy and value networks.
    print("\nInitializing the shared networks")
    env = gym.make("CartPole-v1")
    agent = agents.CartPoleA2C(env, args.n_workers, args.gamma)
    policy_network, value_network = agent.policy_network, agent.value_network
    policy_network.share_memory()
    value_network.share_memory()

    print("\nInitializing the shared optimizers")
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=args.lr_policy)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=args.lr_value)
    share_adam_optimizer(policy_optimizer)
    share_adam_optimizer(value_optimizer)

    print("\nInitializing the processes")
    processes = [mp.Process(target=train_a2c_cartpole, args=(i, 
                                                            policy_network,
                                                            value_network, 
                                                            policy_optimizer,
                                                            value_optimizer, 
                                                            args.gamma,
                                                            args.n_workers,
                                                            args.t_max,
                                                            args.n_train,
                                                            args.alpha_entropy,
                                                            args.thresh,
                                                            args.print_iter,
                                                            args.log_dir,
                                                            barrier, 
                                                            lock, 
                                                            )) for i in range(args.n_workers)]
    
    # Start all processes
    print("\nStarting the processes")
    for p in processes:
        p.start()
    
    # wait for all processes to finish
    for p in processes:
        p.join()

    print(f"\nMain Process: All workers have stopped after {time() - start_time} seconds")

    # Test
    env = gym.make("CartPole-v1", render_mode='human')
    agent.set_env(env)
    agent.policy_network = policy_network
    agent.value_network = value_network
    agent.test(args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif
    if args.path_gif is not None:
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        agent.set_env(env)
        agent.save_gif(args.path_gif, n_episodes=1, duration=150)
        env.close()