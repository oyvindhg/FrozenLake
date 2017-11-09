import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random


class policy_net():
    def __init__(self, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate):

        # Build the neural network
        self.input = tf.placeholder(shape=[None, n_states], dtype=tf.float32)
        dropout_layer = slim.dropout(self.input, 1 - dropout_rate, is_training=True)
        for layer in range(1, len(hidden_layer_size)):
            hidden_layer = slim.fully_connected(dropout_layer, hidden_layer_size[layer], activation_fn=tf.nn.relu)
            dropout_layer = slim.dropout(hidden_layer, 1 - dropout_rate, is_training=True)

        logits = slim.fully_connected(dropout_layer, n_actions)
        self.output = tf.nn.softmax(logits)

        # Select a random action based on the estimated probabilities
        self.p_actions = tf.concat(axis=1, values=[self.output])
        self.select_action = tf.multinomial(tf.log(self.p_actions), num_samples=1)[0][0]

        # Functions for calculating loss
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.cost = -tf.reduce_mean(tf.log(self.responsible_outputs + 1e-20) * self.reward_holder)

        # Functions for calculating the gradients
        tvars = tf.trainable_variables()
        self.hm = tvars
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

def average_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    averaged_r = np.zeros(len(r))
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add + r[t]
        averaged_r[t] = running_add / (len(r) - t)
    return averaged_r

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


def run(env, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate, total_episodes, max_steps, ep_per_update, gamma):

    actor = policy_net(learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate)

    baseline = value_net(learning_rate, n_states, hidden_layer_size)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        total_reward = []

        gradBuffer = sess.run(tf.trainable_variables())
        for i, grad in enumerate(gradBuffer):
            gradBuffer[i] = grad * 0

        avg_rewards = []
        ep_count = 0
        #for ep_count in range(1, total_episodes+1):
        while(True):
            ep_count += 1

            obs = env.reset()

            ep_reward = 0
            obs_history = []
            action_history = []
            reward_history = []
            value_history = []
            for step in range(max_steps):

                obs_history.append(obs)
                action = sess.run(actor.select_action, feed_dict={actor.input: [obs]})

                # If the output (softmax) outputs NAN because all weights are 0, we need to draw randomly.
                # if action == n_actions:
                #     action = random.randint(0,n_actions - 1)
                    #print(action)
                    # gradBuffer = sess.run(tf.trainable_variables())
                    #for i, grad in enumerate(gradBuffer):
                    #    print(grad)

                #lol = [0.0, 0.0]

                #[c+1 for c in lol]

                #print(lol)

                #u = sess.run(actor.output, feed_dict={actor.input: [obs]})

                #print(u)

                #print(action)
                action_history.append(action)
                action_strength = [(action - n_actions / 2) / (n_actions / 4)]

                obs, reward, done, info = env.step(action_strength)

                #reward = action_strength[0]


                if ep_count % 100 == 0:
                    env.render()

                reward_history.append(reward)
                ep_reward += reward

                if done == True:
                    break

            return_history = average_rewards(reward_history)
            obs_history = np.vstack(obs_history)

            value_history = sess.run(baseline.output, feed_dict={baseline.input: obs_history})
            dell_history = return_history - value_history[:][0]

            # for step in range(len(return_history)):
            #     print(obs_history[step][:])
            #     value_history.append(sess.run(baseline.output, feed_dict={baseline.input: [obs_history[step][:]]}))
            #     dell_history.append(return_history[step] - value_history[step])
            #     sess.run(baseline.update_batch,
            #              feed_dict={baseline.input: [obs_history[step][:]], baseline.new_value: [return_history[step]]})

            # print('rewards:', reward_history)
            # print('returns:', return_history)
            # print('value:', value_history)
            # print('dell:', dell_history)

            gradBuffer = sess.run(tf.trainable_variables())
            # for i, grad in enumerate(gradBuffer):
            #     print(grad)


            sess.run(baseline.update_batch, feed_dict={baseline.input: obs_history, baseline.new_value: return_history})

            #I GET IT NOW. LOOK AT SUTTON BARTO. IT IS dell * nabla (value of the state v). Not nabla (dell). Nei, det blir kanskje det samme (kjerneregel)

            disc_dell_history = step_weights(dell_history, gamma)
            norm_dell_history = normalize(dell_history)

            feed_dict={actor.reward_holder: dell_history,
                       actor.action_holder: action_history,
                       actor.input: obs_history}

            # print('dell:', dell_history)
            # print('action:', action_history)
            # print('obs:', obs_history)

            #for i in range(0,len(dell_history)):
                #dell_history[i] = -1
                #action_history[i] = 9

            grads = sess.run(actor.gradients, feed_dict=feed_dict)

            #print('grads:', grads)
            for i, grad in enumerate(grads):
                #clipped_grad = tf.clip_by_norm(grad, 10)
                #a = tf.norm(grad,2)
                #print(a.eval())
                # print('not OK')
                # print('c:', clipped_grad.eval())
                # print('g:', grad)
                gradBuffer[i] += grad


            if ep_count % ep_per_update == 0:
                #print('hmm')
                feed_dict = dict(zip(actor.gradient_holders, gradBuffer))
                sess.run(actor.update_batch, feed_dict=feed_dict)



                for i, grad in enumerate(gradBuffer):
                    gradBuffer[i] = grad * 0


            total_reward.append(ep_reward)

            if ep_count % 10 == 0:
                avg_rewards.append(np.mean(total_reward[-10:]))
                print('reward:', np.mean(total_reward[-10:]))

        return avg_rewards