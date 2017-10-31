import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class policy_net():
    def __init__(self, learning_rate, n_states, action_low, action_high, hidden_layer_size):

        # Build the neural network
        self.input = tf.placeholder(shape=[None, n_states], dtype=tf.float32)
        hidden = slim.stack(self.input, slim.fully_connected, hidden_layer_size, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, 1, activation_fn=tf.nn.sigmoid)

        # Select a random action based on the estimated probabilities
        self.action_avg = (action_high + action_low)/2
        self.mean_action = (self.output*(action_high - action_low) + self.action_avg - 0.5*(action_high - action_low))[0][0]

        # Functions for calculating loss
        self.action_dist_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.cost = -tf.reduce_mean(tf.log(self.mean_action) * self.action_dist_holder)

        # Functions for calculating the gradients
        tvars = tf.trainable_variables()
        self.gradients = tf.gradients(self.cost, tvars)
        self.gradient_holders = []
        for i, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(i) + '_holder')
            self.gradient_holders.append(placeholder)

        # Update using the gradients
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

class value_net():
    def __init__(self, learning_rate, n_states, hidden_layer_size):

        # Build the neural network
        self.input = tf.placeholder(shape=[None, n_states], dtype=tf.float32)
        hidden = slim.stack(self.input, slim.fully_connected, hidden_layer_size, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, 1, activation_fn=None)

        # The error function
        self.new_value = tf.placeholder(shape=[None], dtype=tf.float32)
        self.out = tf.reshape(self.output, [-1])
        self.diffs = self.out - self.new_value
        self.cost = tf.reduce_mean(tf.nn.l2_loss(self.diffs))

        # Functions for calculating the gradients
        tvars = tf.trainable_variables()
        self.gradients = tf.gradients(self.cost, tvars)

        # Update using the gradients
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradients, tvars))


def discount_rewards(r, g):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros(len(r))
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * g + r[t]
        discounted_r[t] = running_add
    return discounted_r

def step_weights(r, g):
    weighted_r = np.zeros(len(r))
    for t in range(0, len(r)):
        weighted_r[t] = r[t] * pow(g, t)
    return weighted_r

def normalize(r):
    mean = r.mean()
    std = r.std()
    normalized_r = (r-mean)/std
    return normalized_r


def run(env, learning_rate, n_states, action_low, action_high, hidden_layer_size, total_episodes, max_steps, ep_per_update, gamma):

    actor = policy_net(learning_rate, n_states, action_low, action_high, hidden_layer_size)

    baseline = value_net(learning_rate, n_states, hidden_layer_size)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        total_reward = []

        gradBuffer = sess.run(tf.trainable_variables())
        for i, grad in enumerate(gradBuffer):
            gradBuffer[i] = grad * 0

        avg_rewards = []

        for ep_count in range(1, total_episodes+1):
        #while(True):

            obs = env.reset()

            std_dev = 1

            ep_reward = 0
            obs_history = []
            obs_history_actor = []
            action_dist_history = []
            action_dist_history_actor = []
            reward_history = []
            for step in range(max_steps):

                obs_history.append(obs)
                # print(obs)
                mean_action = sess.run(actor.mean_action, feed_dict={actor.input: [obs]})


                action = np.random.normal(mean_action, std_dev)
                while action < action_low or action > action_high:
                    action = np.random.normal(mean_action, std_dev)

                action_dist_history.append(action - mean_action)

                # print('mean')
                # print(mean_action)
                # print('action', action)

                obs, reward, done, info = env.step([action])

                if ep_count % 100 == 0:
                    env.render()

                reward_history.append(reward)
                ep_reward += reward

                if done == True:
                    break

            reward_history = discount_rewards(reward_history, gamma)
            obs_history = np.vstack(obs_history)

            return_history = reward_history
            value_history = sess.run(baseline.output, feed_dict={baseline.input: obs_history})
            dell_history = return_history - value_history[:][0]

            sess.run(baseline.update_batch, feed_dict={baseline.input: obs_history, baseline.new_value: return_history})

            update_actor = False
            for i, dell in enumerate(dell_history):
                if dell > 0:
                    action_dist_history_actor.append(action_dist_history[i])
                    obs_history_actor.append(obs_history[i])
                    update_actor = True

            if update_actor:
                feed_dict={actor.action_dist_holder: action_dist_history_actor,
                           actor.input: obs_history_actor}

                grads = sess.run(actor.gradients, feed_dict=feed_dict)

                for i, grad in enumerate(grads):
                    gradBuffer[i] += grad


            if ep_count % ep_per_update == 0: #Mulig feil hvis ingen gradbuffer
                feed_dict = dict(zip(actor.gradient_holders, gradBuffer))
                sess.run(actor.update_batch, feed_dict=feed_dict)


                for i, grad in enumerate(gradBuffer):
                    gradBuffer[i] = grad * 0


            total_reward.append(ep_reward)

            if ep_count % 10 == 0:
                avg_rewards.append(np.mean(total_reward[-10:]))

        return avg_rewards