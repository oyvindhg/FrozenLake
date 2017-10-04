#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

def discount_rewards(r, g):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = [0 for i in range(len(r))]
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * g + r[t]
        discounted_r[t] = running_add
    return discounted_r


def run_pg(env, g):

    import tensorflow as tf
    from tensorflow.contrib.layers import fully_connected

    # 1: Specify the neural network architecture
    n_inputs = 1 # one number can specify the state
    n_hidden = 4 # it's a simple task, we don't need more hidden neurons
    n_outputs = 4 # prob of turning left, up, right and down

    learning_rate = 0.01

    initializer = tf.contrib.layers.variance_scaling_initializer()


    # 2: Build the neural network
    network_input = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = fully_connected(network_input, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
    logits = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer)
    outputs = tf.nn.softmax(logits)


    # 3: Select a random action based on the estimated probabilities
    p_actions = tf.concat(axis=1, values=[outputs])
    sel_action = tf.multinomial(tf.log(p_actions), num_samples=1)[0][0]


    # 4: Define functions for policy gradient
    reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
    action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

    indexes = tf.range(0, tf.shape(outputs)[0]) * tf.shape(outputs)[1] + action_holder

    responsible_outputs = tf.gather(tf.reshape(outputs, [-1]), indexes)

    chosen_output = tf.gather(outputs[0][:], action_holder)

    tvars = tf.trainable_variables()

    loss = -tf.reduce_mean(tf.log(responsible_outputs) * reward_holder) #Create value for action that led to some reward
    gradients = tf.gradients(loss, tvars)

    gradient_holders = []
    for i, var in enumerate(tvars):
        placeholder = tf.placeholder(tf.float32, name=str(i) + '_holder')
        gradient_holders.append(placeholder)


    # 5: Function for getting best output
    best_action = tf.argmax(outputs, 1)


    # 6: Initialize network
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_batch = optimizer.apply_gradients(zip(gradient_holders, tvars))

    init = tf.global_variables_initializer()


    # 7: Training
    total_episodes = 2  # Set total number of episodes to train agent on
    max_steps = 1       # Set maximum number of steps per episode
    ep_per_update = 5   # Train the policy after this number of episodes


    with tf.Session() as sess:
        sess.run(init)

        ep_count = 0
        total_reward = []

        ind = sess.run(responsible_outputs, feed_dict={network_input: [[0]],  action_holder: [0]})
        #print(ind)

        ind2 = sess.run(chosen_output, feed_dict={network_input: [[0]], action_holder: [0]})
        #print(ind2)
        #print('lol')

        some = []
        some.append([2])
        some.append([3])
        some_s = np.vstack(some)
        #print(some_s[:,:])
        #print(some_s.size)

        gradBuffer = sess.run(tf.trainable_variables())
        for i, grad in enumerate(gradBuffer):
            gradBuffer[i] = grad * 0

        print('after')

        while ep_count < total_episodes:

            ep_count += 1
            obs = env.reset()
            ep_reward = 0
            obs_history = []
            action_history = []
            reward_history = []
            for step in range(max_steps):

                obs_history.append(obs)

                action = sess.run(sel_action, feed_dict={network_input: obs.reshape(1, n_inputs)})
                action_history.append(action)

                obs, reward, done, info = env.step(action)
                obs = np.array(obs)

                reward_history.append(reward)

                ep_reward += reward

                if done == True:
                    break

            #Calculate gradients
            reward_history = discount_rewards(reward_history, g)
            obs_history = np.vstack(obs_history)

            feed_dict={reward_holder: reward_history,
                       action_holder: action_history,
                       network_input: obs_history}

            #print(obs_history)
            #print(action_history)
            #print(reward_history)

            gradBuffer = sess.run(tf.trainable_variables())
            for delete_later, grad in enumerate(gradBuffer):
                print(grad)

            grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [1],network_input: [[0]]})

            for i, grad in enumerate(grads):
                gradBuffer[i] += grad
                print(grad)

            feed_dict = dict(zip(gradient_holders, gradBuffer))
            sess.run(update_batch, feed_dict=feed_dict)

            gradBuffer = sess.run(tf.trainable_variables())
            for delete_later, grad in enumerate(gradBuffer):
                print(grad)

            for i, grad in enumerate(gradBuffer):
                gradBuffer[i] = grad * 0


            total_reward.append(ep_reward)

            if ep_count % 100 == 0:
                print(np.mean(total_reward[-100:]))