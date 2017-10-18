import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class network():
    def __init__(self, learning_rate, n_states, n_actions, hidden_layer_size):

        # Build the neural network
        self.input = tf.placeholder(shape=[None, n_states], dtype=tf.float32)
        hidden = slim.stack(self.input, slim.fully_connected, hidden_layer_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, n_actions, activation_fn=tf.nn.softmax, biases_initializer=None)

        # Select a random action based on the estimated probabilities
        self.p_actions = tf.concat(axis=1, values=[self.output])
        self.select_action = tf.multinomial(tf.log(self.p_actions), num_samples=1)[0][0]

        # Functions for calculating loss
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        #Find the best action
        self.best_action = tf.argmax(self.output, 1)

        # Functions for calculating the gradients
        tvars = tf.trainable_variables()
        self.gradients = tf.gradients(self.loss, tvars)
        self.gradient_holders = []
        for i, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(i) + '_holder')
            self.gradient_holders.append(placeholder)

        # Update using the gradients
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


def discount_rewards(r, g, baseline):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros(len(r))
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * g + r[t]
        discounted_r[t] = running_add
    if baseline:
        mean = discounted_r.mean()
        std = discounted_r.std()
        discounted_r = (discounted_r-mean)/std
    return discounted_r


def run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes, max_steps, ep_per_update, gamma):

    actor = network(learning_rate, n_states, n_actions, hidden_layer_size)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        ep_count = 0
        total_reward = []

        gradBuffer = sess.run(tf.trainable_variables())
        for i, grad in enumerate(gradBuffer):
            gradBuffer[i] = grad * 0

        #while ep_count < total_episodes:
        while(True):

            ep_count += 1
            obs = env.reset()

            ep_reward = 0
            state = [0 for i in range(16)]
            state[obs] = 1
            obs_history = []
            action_history = []
            reward_history = []
            for step in range(max_steps):

                obs_history.append(state)
                action = sess.run(actor.select_action, feed_dict={actor.input: [state]})
                action_history.append(action)


                state[obs] = 0
                obs, reward, done, info = env.step(action)

                #if ep_count % 100 == 0:
                    #env.render()

                state[obs] = 1

                reward_history.append(reward)
                ep_reward += reward

                if done == True:
                    break

            reward_history = discount_rewards(reward_history, gamma, baseline=False)
            obs_history = np.vstack(obs_history)

            feed_dict={actor.reward_holder: reward_history,
                       actor.action_holder: action_history,
                       actor.input: obs_history}

            grads = sess.run(actor.gradients, feed_dict=feed_dict)

            for i, grad in enumerate(grads):
                gradBuffer[i] += grad


            if ep_count % ep_per_update == 0 and ep_count != 0:
                feed_dict = dict(zip(actor.gradient_holders, gradBuffer))
                sess.run(actor.update_batch, feed_dict=feed_dict)


                for i, grad in enumerate(gradBuffer):
                    gradBuffer[i] = grad * 0


            total_reward.append(ep_reward)

            if ep_count % 100 == 0:
                print(np.mean(total_reward[-100:]))

                pol = [0 for s in range(4 * 4)]
                pr_state = np.array([0 for i in range(16)])
                for s in range(16):
                    pr_state[s] = 1
                    if s > 0:
                        pr_state[s - 1] = 0
                    pol[s] = sess.run(actor.best_action, feed_dict={actor.input: [pr_state]})

                print('Best action')
                for row in range(4):
                    print(pol[row * 4], pol[row * 4 + 1], pol[row * 4 + 2], pol[row * 4 + 3])

                print('Output prob')
                pr_state = np.array([0 for i in range(16)])
                for s in range(16):
                    pr_state[s] = 1
                    if s > 0:
                        pr_state[s - 1] = 0
                    a = sess.run(actor.output, feed_dict={actor.input: [pr_state]})
                    print(a)