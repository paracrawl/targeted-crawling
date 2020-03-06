#!/usr/bin/env python3
#https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb?source=post_page-----ded33892c724----------------------
#https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

try:
    xrange = xrange
except:
    xrange = range

######################################################################################
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    #print("r", r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        #print("t", t)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

######################################################################################
class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)

        # softmax
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1) # NOT used, instead randomly pick action in CPU according to output probs

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32) #  0 or 1
        
        self.r1 = tf.range(0, tf.shape(self.output)[0])  # 0 1 2 3 4 len=length of trajectory
        self.r2 = tf.shape(self.output)[1]   # a_size = 2     
        self.r3 = self.r1 * self.r2          # 0 2 4 6 8
        self.indexes = self.r3 + self.action_holder # r3 + 0/1 offset depending on action

        self.o1 = tf.reshape(self.output, [-1]) # all action probs in 1-d
        self.responsible_outputs = tf.gather(self.o1, self.indexes) # the prob of the action. Should have just stored it!? len=length of trajectory

        self.l1 = tf.log(self.responsible_outputs)
        self.l2 = self.l1 * self.reward_holder  # log prob * reward. len=length of trajectory
        self.loss = -tf.reduce_mean(self.l2)    # 1 number
        
        # calc grads, but don't actually update weights
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        #for idx,var in enumerate(tvars): # idx = contiguous int, var = variable shape (4,8) (8,2)
        #    print("idx", idx)
        #    print("var", var)
        #    placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
        #    self.gradient_holders.append(placeholder)
        self.gradient_holder0 = tf.placeholder(tf.float32)
        self.gradient_holder1 = tf.placeholder(tf.float32)
        self.gradient_holders.append(self.gradient_holder0)
        self.gradient_holders.append(self.gradient_holder1)

        self.gradients = tf.gradients(self.loss,tvars) # grads same shape as gradient_holder0. (4,8) (8,2)
        
        # update weights
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

######################################################################################
def main():
    env = gym.make('CartPole-v0')

    gamma = 0.99

    tf.reset_default_graph() #Clear the Tensorflow graph.

    myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.

    total_episodes = 5000 #Set total number of episodes to train agent on.
    max_ep = 999
    update_frequency = 5

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        total_length = []
            
        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
            
        while i < total_episodes:
            s = env.reset()
            running_reward = 0
            ep_history = []
            for j in range(max_ep):
                #Probabilistically pick an action given our network outputs.
                a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
                a = np.random.choice(a_dist[0],p=a_dist[0])
                #print("a", a)
                a = np.argmax(a_dist == a)
                #print("a_dist", a, a_dist)

                s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
                ep_history.append([s,a,r,s1])
                s = s1
                running_reward += r
                if d == True:
                    #Update the network.
                    #print("ep_history", len(ep_history))
                    #print("ep_history", ep_history)
                    ep_history = np.array(ep_history)
                    #print("   ep_history", ep_history.shape, ep_history)
                    #print("ep_history", ep_history[:,2])
                    ep_history[:,2] = discount_rewards(ep_history[:,2], gamma)
                    #print("   ep_history", ep_history[:,2].shape, ep_history[:,2])

                    feed_dict={myAgent.reward_holder:   ep_history[:,2],
                               myAgent.action_holder:   ep_history[:,1],
                               myAgent.state_in:        np.vstack(ep_history[:,0])}
                    [grads, indexes, responsible_outputs, r1, r2, r3, output, o1, l2, loss] = sess.run([myAgent.gradients, 
                                                                                myAgent.indexes, 
                                                                                myAgent.responsible_outputs, 
                                                                                myAgent.r1, 
                                                                                myAgent.r2, 
                                                                                myAgent.r3,
                                                                                myAgent.output,
                                                                                myAgent.o1,
                                                                                myAgent.l2,
                                                                                myAgent.loss], feed_dict=feed_dict)
                    #print("grads", grads)
                    #print("output", output.shape, output)
                    #print("r1", r1)
                    #print("r2", r2)
                    #print("r3", r3)
                    #print("action holder", ep_history[:,1].shape, ep_history[:,1])
                    #print("reward_holder holder", ep_history[:,2].shape, ep_history[:,2])
                    #print("indexes", indexes)
                    #print("o1", o1.shape, o1)
                    #print("responsible_outputs", responsible_outputs.shape, responsible_outputs)
                    #print("l2", l2.shape, l2)
                    #print("loss", loss)
                    #for grad in grads:
                    #    print("grad", grad.shape)
                    #print()

                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad         # accumulate gradients

                    if i % update_frequency == 0 and i != 0:
                        # update every 5 episode
                        #print("gradBuffer", gradBuffer)
                        feed_dict= dict(zip(myAgent.gradient_holders, gradBuffer))
                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                        for ix,grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0
                        #print("   gradBuffer", gradBuffer)
                    #else:
                    #    print("nah")

                    total_reward.append(running_reward)
                    total_length.append(j)
                    break

            
                #Update our running tally of scores.
            if i % 100 == 0:
                print(i, np.mean(total_reward[-100:]))
            i += 1

######################################################################################
main()
