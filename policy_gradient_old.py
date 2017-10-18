#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import frozen_lake_data
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
    n_inputs = 16 # one number can specify the state
    n_hidden = 40 # it's a simple task, we don't need more hidden neurons -LIIIIEEEEESS!
    n_hidden2 = 40
    n_hidden3 = 100
    n_outputs = 4 # prob of turning left, up, right and down

    learning_rate = 0.01

    initializer = tf.contrib.layers.variance_scaling_initializer()


    # 2: Build the neural network
    network_input = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = fully_connected(network_input, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
    hidden2 = fully_connected(hidden, n_hidden2, activation_fn=tf.nn.elu, weights_initializer=initializer)
    #hidden3 = fully_connected(hidden2, n_hidden3, activation_fn=tf.nn.elu, weights_initializer=initializer)
    logits = fully_connected(hidden2, n_outputs, activation_fn=None, weights_initializer=initializer)
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
    total_episodes = 100  # Set total number of episodes to train agent on
    max_steps = 1000       # Set maximum number of steps per episode
    ep_per_update = 5   # Train the policy after this number of episodes


    with tf.Session() as sess:

        sess.run(init)

        hmm = sess.run(tf.trainable_variables())
        # for delete_later, grad in enumerate(hmm):
        #     print(grad)

        print('Output prob')
        pr_state = np.array([0 for i in range(16)])
        for s in range(16):
            pr_state[s] = 1
            if s > 0:
                pr_state[s-1] = 0
            a = sess.run(outputs, feed_dict={network_input: np.reshape(pr_state, (1, n_inputs))})
            print(a)

        ep_count = 0
        total_reward = []

        #ind = sess.run(responsible_outputs, feed_dict={network_input: [[0]],  action_holder: [0]})
        #print(ind)

        #ind2 = sess.run(chosen_output, feed_dict={network_input: [[0]], action_holder: [0]})
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

       # print('after')

        inn = 0


        #while ep_count < total_episodes:
        while(True):

            ep_count += 1

            state = [0 for i in range(16)]

            obs = env.reset()
            state[obs] = 1

            ep_reward = 0
            obs_history = []
            action_history = []
            reward_history = []
            state_history = []
            for step in range(max_steps):

                obs_history.append(obs)
                state_history.append(state)

                action = sess.run(sel_action, feed_dict={network_input: np.reshape(state, (1, n_inputs))})
                action_history.append(action)

                state[obs] = 0
                reward = 0
                done = False
                obs2 = frozen_lake_data.M[obs][action]
                if obs2 == 6:
                    reward = 1000
                    done = True
                elif obs2 in [3, 4, 5]:
                    reward = 0
                    done = True
                elif obs2 == 5:
                    reward = 1
                    done = True
                # else:
                #     reward = -1
                #obs2, reward, done, info = env.step(action)
                obs = np.array(obs2)

                state[obs2] = 1

                #print(state)

                reward_history.append(reward)

                ep_reward += reward

                if done == True:
                    break

            #if ep_reward > 0:
            #    print('hurray!')

            #Calculate gradients
            reward_history = discount_rewards(reward_history, g)
            state_history = np.vstack(state_history)

            feed_dict={reward_holder: reward_history,
                       action_holder: action_history,
                       network_input: state_history}

            if ep_reward > 0:
                print('reward:', reward_history)
                print('actions:', action_history)
                print('obs:', obs_history)


            grads = sess.run(gradients, feed_dict=feed_dict)

            # sss = [0 for s in range(4 * 4)]
            # if inn == 0:
            #     sss[0] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [0], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 1:
            #     sss[1] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [3], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 2:
            #     sss[3] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [3], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 3:
            #     sss[4] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [0], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 4:
            #     sss[5] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [0], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 5:
            #     sss[7] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [0], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 6:
            #     sss[8] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [3], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 7:
            #     sss[9] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [1], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 8:
            #     sss[10] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [0], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 9:
            #     sss[13] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [2], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 10:
            #     sss[14] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [1], network_input: np.reshape(sss, (1, n_inputs))})
            # elif inn == 11:
            #     sss[2] = 1
            #     grads = sess.run(gradients, feed_dict={reward_holder: [1], action_holder: [0], network_input: np.reshape(sss, (1, n_inputs))})
            #
            #
            # inn += 1
            # inn = inn % 12

            #lss = sess.run(loss, feed_dict={reward_holder: [1], action_holder: [1], network_input: [[0]]})
            #print('loss:')
            #print(lss)

            #a = tf.gradients(4, [7,0,0,0])
            #print('a:')
            #print(a)

            for i, grad in enumerate(grads):
                #print(gradBuffer[i])
                gradBuffer[i] += grad
                #print(grad)

            if ep_count % ep_per_update == 0 and ep_count != 0:

                feed_dict = dict(zip(gradient_holders, gradBuffer))

                #feed_dict = dict(zip(gradient_holders, grads))

                sess.run(update_batch, feed_dict=feed_dict)

                #gradBuffer = sess.run(tf.trainable_variables())
                #for delete_later, grad in enumerate(gradBuffer):
                    #print(grad)

                for i, grad in enumerate(gradBuffer):
                    gradBuffer[i] = grad * 0


            total_reward.append(ep_reward)

            if ep_count % 10 == 0:
                print(sum(total_reward[-10:]))

                print('Output prob')
                pr_state = np.array([0 for i in range(16)])
                for s in range(16):
                    pr_state[s] = 1
                    if s > 0:
                        pr_state[s - 1] = 0
                    a = sess.run(outputs, feed_dict={network_input: np.reshape(pr_state, (1, n_inputs))})
                    print(a)

                #print('now new')
                #hmm = sess.run(tf.trainable_variables())
                #for delete_later, grad in enumerate(hmm):
                    #print(grad)

                pol = [0 for s in range(4 * 4)]
                pr_state = np.array([0 for i in range(16)])
                for s in range(16):
                    pr_state[s] = 1
                    if s > 0:
                        pr_state[s - 1] = 0
                    pol[s] = sess.run(best_action, feed_dict = {network_input: np.reshape(pr_state, (1, n_inputs))})

                print('Best action')
                for row in range(4):
                    print(pol[row * 4], pol[row * 4 + 1], pol[row * 4 + 2], pol[row * 4 + 3])
