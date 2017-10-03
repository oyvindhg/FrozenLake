import gym
import policy_iteration
import value_iteration
import Q_learning
import policy_gradient
import plot

if __name__ == "__main__":

    #plot.plot_board(4, 4, 0, [5, 7, 11, 12], 15)

    #g is the discount factor
    g = 0.9

    env = gym.make('FrozenLake-v0')

    policy_gradient.run_pg(env, g)

    #Q_learning.run_q(env, g)
    #policy_iteration.run_pi(env, g)
    #value_iteration.run_vi(env, g)



