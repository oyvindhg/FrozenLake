import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class policy_net():
    def __init__(self, learning_rate, n_states, n_actions, hidden_layer_size):

        # Build the neural network
        self.input = tf.placeholder(shape=[None, n_states], dtype=tf.float32)
        hidden = slim.stack(self.input, slim.fully_connected, hidden_layer_size, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, n_actions, activation_fn=tf.nn.softmax)

        # Select a random action based on the estimated probabilities
        self.p_actions = tf.concat(axis=1, values=[self.output])
        self.select_action = tf.multinomial(tf.log(self.p_actions), num_samples=1)[0][0]

        # Functions for calculating loss
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.cost = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

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


def run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes, max_steps, ep_per_update, gamma):

    actor = policy_net(learning_rate, n_states, n_actions, hidden_layer_size)

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

            ep_reward = 0
            obs_history = []
            action_history = []
            reward_history = []
            value_history = []
            for step in range(max_steps):

                obs_history.append(obs)
                action = sess.run(actor.select_action, feed_dict={actor.input: [obs]})
                action_history.append(action)

                obs, reward, done, info = env.step(action)
                # if ep_count % 100 == 0:
                #     env.render()

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

            disc_dell_history = step_weights(dell_history, gamma)
            norm_dell_history = normalize(dell_history)

            feed_dict={actor.reward_holder: norm_dell_history,
                       actor.action_holder: action_history,
                       actor.input: obs_history}

            grads = sess.run(actor.gradients, feed_dict=feed_dict)



            for i, grad in enumerate(grads):
                gradBuffer[i] += grad


            if ep_count % ep_per_update == 0:
                feed_dict = dict(zip(actor.gradient_holders, gradBuffer))
                sess.run(actor.update_batch, feed_dict=feed_dict)


                for i, grad in enumerate(gradBuffer):
                    gradBuffer[i] = grad * 0


            total_reward.append(ep_reward)

            if ep_count % 10 == 0:
                avg_rewards.append(np.mean(total_reward[-10:]))

        return avg_rewards