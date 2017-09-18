import gym
import policy_iteration

if __name__ == "__main__":

    #g is the discount factor
    g = 0.9

    V, pol = policy_iteration.policy_iteration(g)
    #Q, pol = iteration.value_iteration(g)

    env = gym.make('FrozenLake-v0')

    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = pol[observation]
            observation, reward, done, info = env.step(action)
            if done:
                env.render()
                print("Episode finished after {} timesteps".format(t + 1))
                break