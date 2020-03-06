#!/usr/bin/env python3

import random
from collections import namedtuple

import numpy as np
import pylab as plt
import tensorflow as tf


######################################################################################
class LearningParams:
    def __init__(self):
        self.gamma = 1  # 0.99
        self.lrn_rate = 0.1
        self.q_lrn_rate = 1
        self.max_epochs = 100001
        self.eps = 1  # 0.7
        self.maxBatchSize = 32
        self.debug = False
        self.walk = 1000


######################################################################################
class Qnetwork():
    def __init__(self, lrn_rate, env):
        # These lines establish the feed-forward part of the network used to choose actions
        EMBED_DIM = 80

        NUM_ACTIONS = 5
        INPUT_DIM = EMBED_DIM // NUM_ACTIONS

        HIDDEN_DIM = 128

        # EMBEDDINGS
        self.embeddings = tf.Variable(tf.random_uniform([env.ns, INPUT_DIM], 0, 0.01))

        self.input = tf.placeholder(shape=[None, NUM_ACTIONS], dtype=tf.int32)

        self.embedding = tf.nn.embedding_lookup(self.embeddings, self.input)
        self.embedding = tf.reshape(self.embedding, [tf.shape(self.input)[0], EMBED_DIM])

        # HIDDEN 1
        self.hidden1 = self.embedding

        self.Whidden1 = tf.Variable(tf.random_uniform([EMBED_DIM, EMBED_DIM], 0, 0.01))
        self.hidden1 = tf.matmul(self.hidden1, self.Whidden1)

        # self.BiasHidden1 = tf.Variable(tf.random_uniform([1, EMBED_DIM], 0, 0.01))
        # self.hidden1 = tf.add(self.hidden1, self.BiasHidden1)

        self.hidden1 = tf.math.l2_normalize(self.hidden1, axis=1)
        # self.hidden1 = tf.nn.relu(self.hidden1)

        # HIDDEN 2
        self.hidden2 = self.hidden1

        self.Whidden2 = tf.Variable(tf.random_uniform([EMBED_DIM, HIDDEN_DIM], 0, 0.01))

        self.hidden2 = tf.matmul(self.hidden2, self.Whidden2)

        self.BiasHidden2 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))

        self.hidden2 = tf.add(self.hidden2, self.BiasHidden2)

        # OUTPUT
        self.Wout = tf.Variable(tf.random_uniform([HIDDEN_DIM, 5], 0, 0.01))

        self.Qout = tf.matmul(self.hidden2, self.Wout)

        self.predict = tf.argmax(self.Qout, 1)

        self.sumWeight = tf.reduce_sum(self.Wout) \
                         + tf.reduce_sum(self.BiasHidden2) \
                         + tf.reduce_sum(self.Whidden2) \
                         + tf.reduce_sum(self.Whidden1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[None, 5], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
        self.trainer = tf.train.AdamOptimizer()  # learning_rate=lrn_rate)

        self.updateModel = self.trainer.minimize(self.loss)

    def PrintQ(self, curr, env, sess):
        # print("hh", next, hh)
        neighbours = env.GetNeighBours(curr)
        a, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: neighbours})
        # print("curr=", curr, "a=", a, "allQ=", allQ, neighbours)
        print("curr=", curr, allQ, neighbours)

    def PrintAllQ(self, env, sess):
        print("         Q-values                          Next state")
        for curr in range(env.ns):
            self.PrintQ(curr, env, sess)


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
        # self.F[14, 9] = 1;

        for i in range(self.ns):
            self.F[i, self.ns - 1] = 1
        # print("F", self.F)

    def GetNextState(self, curr, action, neighbours):
        # print("curr", curr, action, neighbours)
        next = neighbours[0, action]
        assert (next >= 0)
        # print("next", next)

        done = False
        if next == self.goal:
            reward = 8.5
            done = True
        elif next == self.ns - 1:
            reward = 0
            done = True
        else:
            reward = -1

        return next, reward, done

    def GetNeighBours(self, curr):
        col = self.F[curr, :]
        ret = []
        for i in range(len(col)):
            if col[i] == 1:
                ret.append(i)

        for i in range(len(ret), 5):
            ret.append(self.ns - 1)

        random.shuffle(ret)

        # ret = np.empty([5,1])
        ret = np.array(ret)
        ret = ret.reshape([1, 5])
        # print("GetNeighBours", ret.shape, ret)

        return ret

    def Walk(self, start, sess, qn, printQ):
        curr = start
        i = 0
        totReward = 0
        print(str(curr) + "->", end="")
        while True:
            # print("curr", curr)
            # print("hh", next, hh)
            neighbours = self.GetNeighBours(curr)
            action, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.input: neighbours})
            action = action[0]
            next, reward, done = self.GetNextState(curr, action, neighbours)
            totReward += reward

            # if printQ:
            #    print("printQ", action, allQ, neighbours)

            # print("(" + str(action) + ")", str(next) + "(" + str(reward) + ") -> ", end="")
            print(str(next) + "->", end="")
            curr = next

            if done: break
            if curr == self.goal: break

            i += 1
            if i > 20:
                print("LOOPING", end="")
                break

        print(" ", totReward)

    def WalkAll(self, sess, qn):
        for start in range(self.ns):
            self.Walk(start, sess, qn, False)


######################################################################################

def Neural(epoch, curr, params, env, sess, qn):
    # NEURAL
    # print("hh", next, hh)
    neighbours = env.GetNeighBours(curr)
    a, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.input: neighbours})
    a = a[0]
    if np.random.rand(1) < params.eps:
        a = np.random.randint(0, 5)

    next, r, done = env.GetNextState(curr, a, neighbours)
    # print("curr=", curr, "a=", a, "next=", next, "r=", r, "allQ=", allQ)

    # Obtain the Q' values by feeding the new state through our network
    if curr == env.ns - 1:
        targetQ = np.zeros([1, 5])
        maxQ1 = 0
    else:
        # print("  hh2", hh2)
        nextNeighbours = env.GetNeighBours(next)
        Q1 = sess.run(qn.Qout, feed_dict={qn.input: nextNeighbours})
        # print("  Q1", Q1)
        maxQ1 = np.max(Q1)

        # targetQ = allQ
        targetQ = np.array(allQ, copy=True)
        # print("  targetQ", targetQ)
        newVal = r + params.gamma * maxQ1
        # targetQ[0, a] = (1 - params.q_lrn_rate) * targetQ[0, a] + params.q_lrn_rate * newVal
        targetQ[0, a] = newVal
        # print("  targetQ", targetQ)

    # print("  targetQ", targetQ, maxQ1)
    # print("  new Q", a, allQ)

    Transition = namedtuple("Transition", "curr next done neighbours targetQ")
    transition = Transition(curr, next, done, np.array(neighbours, copy=True), np.array(targetQ, copy=True))

    return transition


def UpdateQN(params, env, sess, qn, neighbours, targetQ):
    outLoop = 1000
    if params.debug:
        outLoop = 1
        print("neighbours", neighbours)
        print("targetQ", targetQ)
        print()

    if params.debug:
        outs = [qn.updateModel, qn.loss, qn.sumWeight, qn.Wout, qn.Whidden2, qn.BiasHidden2, qn.Qout, qn.embeddings,
                qn.embedding]
        _, loss, sumWeight, Wout, Whidden, BiasHidden, Qout, embeddings, embedding = sess.run(outs,
                                                                                              feed_dict={
                                                                                                  qn.input: neighbours,
                                                                                                  qn.nextQ: targetQ})
        # print("embeddings", embeddings.shape, embeddings)
        # print("embedding", embedding.shape, embedding)
        # print("embedConcat", embedConcat.shape)

        # print("  Wout\n", Wout)
        # print("  Whidden\n", Whidden)
        # print("  BiasHidden\n", BiasHidden)

        # print("curr", curr, "next", next, "action", a)
        # print("allQ", allQ)
        # print("targetQ", targetQ)
        # print("Qout", Qout)
        # print("eps", params.eps)

    else:
        _, loss, sumWeight = sess.run([qn.updateModel, qn.loss, qn.sumWeight],
                                      feed_dict={qn.input: neighbours, qn.nextQ: targetQ})

    # print("loss", loss)
    return loss, sumWeight


def Trajectory(epoch, curr, params, env, sess, qn):
    path = []
    while (True):
        transition = Neural(epoch, curr, params, env, sess, qn)
        path.append(transition)
        curr = transition.next

        if transition.done: break

    trajSize = len(path)
    trajNeighbours = np.empty([trajSize, 5], dtype=np.int)
    trajTargetQ = np.empty([trajSize, 5])

    i = 0
    for transition in path:
        # print("transition", transition.neighbours.shape, transition.targetQ.shape)
        trajNeighbours[i, :] = transition.neighbours
        trajTargetQ[i, :] = transition.targetQ

        i += 1
    return curr, path, trajNeighbours, trajTargetQ


def ExpandCorpus(corpusNeighbours, corpusTargetQ, trajNeighbours, trajTargetQ):
    corpusNeighbours = np.append(corpusNeighbours, trajNeighbours, axis=0)
    corpusTargetQ = np.append(corpusTargetQ, trajTargetQ, axis=0)
    return corpusNeighbours, corpusTargetQ


def CreateBatch(corpusNeighbours, corpusTargetQ, batchSize):
    batchNeighbours = corpusNeighbours[0:batchSize, :]
    batchTargetQ = corpusTargetQ[0:batchSize, :]
    corpusNeighbours = corpusNeighbours[batchSize:, :]
    corpusTargetQ = corpusTargetQ[batchSize:, :]

    return batchNeighbours, batchTargetQ, corpusNeighbours, corpusTargetQ


def Train(params, env, sess, qn):
    losses = []
    sumWeights = []

    corpusNeighbours = np.empty([0, 5], dtype=np.int)
    corpusTargetQ = np.empty([0, 5])

    for epoch in range(params.max_epochs):
        startState = np.random.randint(0, env.ns)  # random start state
        stopState, path, trajNeighbours, trajTargetQ = Trajectory(epoch, startState, params, env, sess, qn)

        assert (trajNeighbours.shape[0] == trajTargetQ.shape[0])
        assert (5 == trajNeighbours.shape[1] == trajTargetQ.shape[1])
        trajSize = trajNeighbours.shape[0]

        corpusNeighbours, corpusTargetQ = ExpandCorpus(corpusNeighbours, corpusTargetQ, trajNeighbours, trajTargetQ)
        corpusSize = corpusNeighbours.shape[0]

        if corpusSize >= params.maxBatchSize:
            # print("corpusSize", corpusSize)

            batchNeighbours, batchTargetQ, corpusNeighbours, corpusTargetQ = CreateBatch(corpusNeighbours,
                                                                                         corpusTargetQ,
                                                                                         params.maxBatchSize)
            # print("batchSize", batchNeighbours.shape)
            # print("corpusNeighbours", corpusNeighbours.shape)
            # print("corpusTargetQ", corpusTargetQ.shape)

            loss, sumWeight = UpdateQN(params, env, sess, qn, batchNeighbours, batchTargetQ)
            losses.append(loss)
            sumWeights.append(sumWeight)

        if epoch % params.walk == 0:
            print("\nepoch", epoch)
            qn.PrintAllQ(env, sess)
            env.WalkAll(sess, qn)
            # env.Walk(9, sess, qn, True)

        # add to batch
        if stopState == env.goal:
            # eps = 1. / ((i/50) + 10)
            params.eps *= .999
            params.eps = max(0.1, params.eps)
            # print("eps", params.eps)

            # params.q_lrn_rate * 0.999
            # params.q_lrn_rate = max(0.1, params.q_lrn_rate)
            # print("q_lrn_rate", params.q_lrn_rate)

    # LAST BATCH
    # corpusSize = corpusNeighbours.shape[0]
    # if corpusSize > 0:
    #    UpdateQN(params, env, sess, qn, corpusNeighbours, corpusTargetQ)

    return losses, sumWeights


######################################################################################

######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # =============================================================
    env = Env()

    params = LearningParams()

    tf.reset_default_graph()
    qn = Qnetwork(params.lrn_rate, env)
    init = tf.global_variables_initializer()
    print("qn.Qout", qn.Qout)

    with tf.Session() as sess:
        sess.run(init)

        losses, sumWeights = Train(params, env, sess, qn)
        print("Trained")

        qn.PrintAllQ(env, sess)
        env.WalkAll(sess, qn)

        plt.plot(losses)
        plt.show()

        plt.plot(sumWeights)
        plt.show()

    print("Finished")


if __name__ == "__main__":
    Main()
