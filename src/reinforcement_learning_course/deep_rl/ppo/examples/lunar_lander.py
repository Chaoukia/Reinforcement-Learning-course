import argparse
import multiprocessing as mp
import numpy as np
import torch
import gymnasium as gym
from reinforcement_learning_course.deep_rl.ppo.examples import agents
from reinforcement_learning_course.deep_rl.ppo.algorithms import train_ppo_worker, share_adam_optimizer
from time import time


def train_ppo_cartpole(worker_id, 
                    shared_advantages,
                    advantage_mean, 
                    advantage_std,
                    n_advantages_total,
                    actor_network,
                    critic_network, 
                    policy_optimizer,
                    value_optimizer, 
                    epsilon, 
                    lambd,
                    gamma,
                    n_workers,
                    t_max,
                    n_episodes,
                    alpha_entropy,
                    epochs,
                    thresh,
                    print_iter,
                    log_dir,
                    barrier, 
                    lock, 
                    ) -> None:
    
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=0.0, turbulence_power=0.0)
    agent = agents.LunarLanderPPO(env, n_workers, epsilon, lambd, gamma)
    train_ppo_worker(
        worker_id, 
        shared_advantages,
        advantage_mean, 
        advantage_std,
        n_advantages_total,
        actor_network,
        critic_network, 
        policy_optimizer, 
        value_optimizer, 
        env,
        agent,
        t_max,
        n_episodes,
        alpha_entropy,
        epochs,
        thresh,
        print_iter,
        log_dir,
        barrier, 
        lock, 
    )


if __name__ == '__main__':

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    argparser = argparse.ArgumentParser(description='Parse options')

    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--n_workers', type=int, default=10, help="Number of workers.")
    argparser.add_argument('--epsilon', type=float, default=0.2, help="Epsilon clipping parameter for the PPO policy loss.")
    argparser.add_argument('--lambd', type=float, default=0.95, help="Lambda parameter for calculating the GAE.")
    argparser.add_argument('--n_train', type=int, default=10000, help="Number of training episodes.")
    argparser.add_argument('--t_max', type=int, default=128, help="Number of steps between two consecutive gradient updates.")
    argparser.add_argument('--lr_actor', type=float, default=1e-4, help="Learning rate for the policy network.")
    argparser.add_argument('--lr_critic', type=float, default=1e-4, help="Learning rate for the value network.")
    argparser.add_argument('--alpha_entropy', type=float, default=1.0, help="Penalty on the entropy loss term.")
    argparser.add_argument('--epochs', type=int, default=4, help="Number of epochs.")
    argparser.add_argument('--thresh', type=float, default=250, help="Minimum mean reward to stop training.")
    argparser.add_argument('--log_dir', type=str, default='runs', help="Path where to log the tensorboard runs.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--print_iter', type=int, default=100, help="If 1, save gif of the tested agent, 0 otherwise.")
    argparser.add_argument('--path_weights', type=str, help="Path where to save the actor and critic weights")
    argparser.add_argument('--path_gif', type=str, help="Path where to save the test gif. If not provided, no gif save will be performed.")
    
    args = argparser.parse_args()

    barrier = mp.Barrier(args.n_workers)
    lock = mp.Lock()
    start_time = time()

    # Initilizing the shared parameters of the policy and value networks.
    print("\nInitializing the shared networks")
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=0.0, turbulence_power=0.0)
    agent = agents.LunarLanderPPO(env, args.n_workers, args.epsilon, args.lambd, args.gamma)
    actor_network, critic_network = agent.actor_network, agent.critic_network
    actor_network.share_memory()
    critic_network.share_memory()

    print("\nInitializing the shared optimizers")
    policy_optimizer = torch.optim.Adam(actor_network.parameters(), lr=args.lr_actor)
    value_optimizer = torch.optim.Adam(critic_network.parameters(), lr=args.lr_critic)
    share_adam_optimizer(policy_optimizer)
    share_adam_optimizer(value_optimizer)

    # Initializing the shared array of advantages, the advantage mean, std, and total number gathered.
    print("\nInitializing the shared advantages")
    advantage_mean = mp.Value('d', 0.0)
    advantage_std = mp.Value('d', 0.0)
    n_advantages_total = mp.Value('d', 0.0)
    shared_advantages = mp.Array('d', [0 for i in range(args.n_workers*args.t_max)])
    shared_advantages = np.frombuffer(shared_advantages.get_obj(), dtype=float).reshape((args.n_workers, args.t_max))

    print("\nInitializing the processes")
    processes = [mp.Process(target=train_ppo_cartpole, args=(i, 
                                                            shared_advantages,
                                                            advantage_mean, 
                                                            advantage_std,
                                                            n_advantages_total,
                                                            actor_network,
                                                            critic_network, 
                                                            policy_optimizer,
                                                            value_optimizer, 
                                                            args.epsilon, 
                                                            args.lambd,
                                                            args.gamma,
                                                            args.n_workers,
                                                            args.t_max,
                                                            args.n_train,
                                                            args.alpha_entropy,
                                                            args.epochs,
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
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=0.0, turbulence_power=0.0, render_mode='human')
    agent.set_env(env)
    agent.actor_network = actor_network
    agent.critic_network = critic_network
    agent.test(args.n_test, verbose=args.verbose)
    env.close()

    # Save weights
    if args.path_weights:
        print("Saving the weights")
        agent.save(args.path_weights)
    
    # Save gif
    if args.path_gif is not None:
        print("Saving the test gif")
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=0.0, turbulence_power=0.0, render_mode='rgb_array')
        agent.set_env(env)
        agent.save_gif(args.path_gif, n_episodes=1, duration=150)
        env.close()