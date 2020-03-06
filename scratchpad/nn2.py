#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


######################################################################################
class LearningParams:
    def __init__(self):
        self.gamma = 0.99  # 0.1
        self.lrn_rate = 0.1
        self.max_epochs = 50001
        self.eps = 1  # 0.7


######################################################################################
class Qnetwork():
    def __init__(self, lrn_rate, env):
        # These lines establish the feed-forward part of the network used to choose actions
        HIDDEN_DIM = 128

        self.inputs = tf.placeholder(shape=[1, env.ns], dtype=tf.float32)
        self.hidden = self.inputs

        self.Whidden = tf.Variable(tf.random_uniform([env.ns, HIDDEN_DIM], 0, 0.01))
        # self.Whidden = tf.nn.softmax(self.Whidden, axis=1)
        # self.Whidden = tf.nn.sigmoid(self.Whidden)
        # self.Whidden = tf.math.l2_normalize(self.Whidden, axis=1)
        self.hidden = tf.matmul(self.hidden, self.Whidden)

        # self.Whidden = tf.Variable(tf.random_uniform([1, env.ns], 0, 0.01))
        # self.Whidden = tf.nn.softmax(self.Whidden, axis=1)
        # self.Whidden = tf.nn.sigmoid(self.Whidden)
        # self.Whidden = tf.clip_by_value(self.Whidden, -10, 10)
        # self.hidden = tf.multiply(self.hidden, self.Whidden)

        self.BiasHidden = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        # self.BiasHidden = tf.nn.softmax(self.BiasHidden, axis=1)
        # self.BiasHidden = tf.nn.sigmoid(self.BiasHidden)
        # self.BiasHidden = tf.math.l2_normalize(self.BiasHidden, axis=1)
        self.hidden = tf.add(self.hidden, self.BiasHidden)

        self.W = tf.Variable(tf.random_uniform([HIDDEN_DIM, 5], 0, 0.01))
        # self.W = tf.math.l2_normalize(self.W, axis=1)
        # self.W = tf.nn.sigmoid(self.W)
        # self.W = tf.math.multiply(self.W, 2)
        # self.W = tf.clip_by_value(self.W, -10, 10)

        self.Qout = tf.matmul(self.hidden, self.W)
        # self.Qout = tf.clip_by_value(self.Qout, -10, 10)
        # self.Qout = tf.nn.sigmoid(self.Qout)
        self.Qout = tf.math.multiply(self.Qout, 0.1)

        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1, 5], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
        self.updateModel = self.trainer.minimize(self.loss)

    def my_print1(self, curr, env, sess):
        curr_1Hot = Int2Arrray(curr, env.ns)
        # print("hh", next, hh)
        a, allQ = sess.run([self.predict, self.Qout], feed_dict={self.inputs: curr_1Hot})
        # print("curr=", curr, "a=", a, "allQ=", allQ)
        print(curr, allQ)

    def my_print(self, env, sess):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        np.linspace(2.0, 3.0, num=5)
        print("     stop   up  right  down left")
        for curr in range(env.ns):
            self.my_print1(curr, env, sess)


######################################################################################
# helpers
class Env:
    def __init__(self):
        self.goal = 14
        self.ns = 16  # number of states

        self.F = np.zeros(shape=[self.ns, self.ns], dtype=np.int)  # Feasible
        self.F[0, 1] = 1;
        self.F[0, 5] = 1;
        self.F[1, 0] = 1;
        self.F[2, 3] = 1;
        self.F[3, 2] = 1
        self.F[3, 4] = 1;
        self.F[3, 8] = 1;
        self.F[4, 3] = 1;
        self.F[4, 9] = 1;
        self.F[5, 0] = 1
        self.F[5, 6] = 1;
        self.F[5, 10] = 1;
        self.F[6, 5] = 1;
        # self.F[6, 7] = 1; # hole
        # self.F[7, 6] = 1; # hole
        self.F[7, 8] = 1;
        self.F[7, 12] = 1
        self.F[8, 3] = 1;
        self.F[8, 7] = 1;
        self.F[9, 4] = 1;
        self.F[9, 14] = 1;
        self.F[10, 5] = 1
        self.F[10, 11] = 1;
        self.F[11, 10] = 1;
        self.F[11, 12] = 1;
        self.F[12, 7] = 1;
        self.F[12, 11] = 1;
        self.F[12, 13] = 1;
        self.F[13, 12] = 1;

        for i in range(self.ns):
            self.F[i, self.ns - 1] = 1
        # print("F", self.F)

    def GetNextState(self, curr, action):
        if action == 1:
            next = curr - 5
        elif action == 2:
            next = curr + 1
        elif action == 3:
            next = curr + 5
        elif action == 4:
            next = curr - 1
        elif action == 0:
            next = self.ns - 1
        # assert(next >= 0)
        # print("next", next)

        die = False
        if next < 0 or next >= self.ns or self.F[curr, next] == 0:
            # disallowed actions
            next = self.ns - 1
            reward = -10
            die = True
        elif next == self.goal:
            reward = 8.5
            die = True
        elif action == 0:
            assert (next != self.goal)
            reward = 0
            die = True
        else:
            reward = -1

        return next, reward, die

    def get_poss_next_actions(self, s):
        actions = []
        actions.append(0)
        actions.append(1)
        actions.append(2)
        actions.append(3)
        actions.append(4)

        # print("  actions", actions)
        return actions

    def Walk(self, start, sess, qn):
        curr = start
        i = 0
        totReward = 0
        print(str(curr) + "->", end="")
        while True:
            # print("curr", curr)
            curr_1Hot = Int2Arrray(curr, self.ns)
            # print("hh", next, hh)
            action, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs: curr_1Hot})
            action = action[0]
            next, reward, die = self.GetNextState(curr, action)
            totReward += reward

            print("(" + str(action) + ")", str(next) + "(" + str(reward) + ") -> ", end="")
            # print(str(next) + "->", end="")
            curr = next

            if die: break
            if curr == self.goal: break

            i += 1
            if i > 50:
                print("LOOPING")
                break

        print("done", totReward)


######################################################################################
def Int2Arrray(num, size):
    ret = np.identity(size)[num:num + 1]
    return ret

    str = np.binary_repr(num).zfill(size)
    l = list(str)
    ret = np.array(l, ndmin=2).astype(np.float)
    # print("num", num, ret)
    return ret


def Neural(epoch, curr, params, env, sess, qn):
    # NEURAL
    curr_1Hot = Int2Arrray(curr, env.ns)
    # print("hh", next, hh)
    a, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.inputs: curr_1Hot})
    a = a[0]
    if np.random.rand(1) < params.eps:
        a = np.random.randint(0, 5)

    next, r, die = env.GetNextState(curr, a)
    # print("curr=", curr, "a=", a, "next=", next, "r=", r, "allQ=", allQ)

    # Obtain the Q' values by feeding the new state through our network
    if curr == env.ns - 1:
        targetQ = np.zeros([1, 5])
        maxQ1 = 0
    else:
        next1Hot = Int2Arrray(next, env.ns)
        # print("  hh2", hh2)
        Q1 = sess.run(qn.Qout, feed_dict={qn.inputs: next1Hot})
        # print("  Q1", Q1)
        maxQ1 = np.max(Q1)

        # targetQ = allQ
        targetQ = np.array(allQ, copy=True)
        # print("  targetQ", targetQ)
        targetQ[0, a] = r + params.gamma * maxQ1
        # print("  targetQ", targetQ)

    # print("  targetQ", targetQ, maxQ1)

    if epoch % 10000 == 0:
        outs = [qn.updateModel, qn.W, qn.Whidden, qn.BiasHidden, qn.Qout, qn.inputs]
        _, W, Whidden, BiasHidden, Qout, inputs = sess.run(outs,
                                                           feed_dict={qn.inputs: curr_1Hot, qn.nextQ: targetQ})
        print("epoch", epoch)
        # print("  W\n", W)
        # print("  Whidden\n", Whidden)
        # print("  BiasHidden\n", BiasHidden)
        qn.my_print(env, sess)

        # print("curr", curr, "next", next, "action", a)
        # print("allQ", allQ)
        # print("targetQ", targetQ)
        # print("Qout", Qout)

        print()
    else:
        _, W1 = sess.run([qn.updateModel, qn.W], feed_dict={qn.inputs: curr_1Hot, qn.nextQ: targetQ})

    # print("  new Q", a, allQ)

    return next, die


def Trajectory(epoch, curr, params, env, sess, qn):
    while (True):
        next, done = Neural(epoch, curr, params, env, sess, qn)
        # next, done = Tabular(curr, Q, gamma, lrn_rate, env)
        curr = next

        if done: break
    # print()


def Train(params, env, sess, qn):
    scores = []

    for epoch in range(params.max_epochs):
        curr = np.random.randint(0, env.ns)  # random start state
        Trajectory(epoch, curr, params, env, sess, qn)

        # eps = 1. / ((i/50) + 10)
        # eps *= .99
        # print("eps", eps)

    return scores


######################################################################################

######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print("Setting up maze in memory")

    # =============================================================
    print("Analyzing maze with RL Q-learning")
    env = Env()

    params = LearningParams()

    tf.reset_default_graph()
    qn = Qnetwork(params.lrn_rate, env)
    init = tf.initialize_all_variables()
    print("qn.Qout", qn.Qout)

    with tf.Session() as sess:
        sess.run(init)

        scores = Train(params, env, sess, qn)
        print("Trained")

        qn.my_print(env, sess)

        for start in range(0, env.ns):
            env.Walk(start, sess, qn)

        # plt.plot(scores)
        # plt.show()

        print("Finished")


if __name__ == "__main__":
    Main()
