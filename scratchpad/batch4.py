#!/usr/bin/env python3

import argparse
import configparser
import random
from collections import namedtuple

import mysql.connector
import numpy as np
import pylab as plt
import tensorflow as tf


######################################################################################
def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)


######################################################################################
class LearningParams:
    def __init__(self):
        self.gamma = 1  # 0.99
        self.lrn_rate = 0.1
        self.q_lrn_rate = 1
        self.max_epochs = 30001
        self.eps = 1  # 0.7
        self.maxBatchSize = 32
        self.debug = False
        self.walk = 1000
        self.NUM_ACTIONS = 10


######################################################################################
class Qnetwork():
    def __init__(self, params, env):
        # These lines establish the feed-forward part of the network used to choose actions
        EMBED_DIM = 90

        INPUT_DIM = EMBED_DIM // params.NUM_ACTIONS

        HIDDEN_DIM = 128

        # EMBEDDINGS
        self.embeddings = tf.Variable(tf.random_uniform([env.ns, INPUT_DIM], 0, 0.01))

        self.input = tf.placeholder(shape=[None, params.NUM_ACTIONS], dtype=tf.int32)

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
        self.Wout = tf.Variable(tf.random_uniform([HIDDEN_DIM, params.NUM_ACTIONS], 0, 0.01))

        self.Qout = tf.matmul(self.hidden2, self.Wout)

        self.predict = tf.argmax(self.Qout, 1)

        self.sumWeight = tf.reduce_sum(self.Wout) \
                         + tf.reduce_sum(self.BiasHidden2) \
                         + tf.reduce_sum(self.Whidden2) \
                         + tf.reduce_sum(self.Whidden1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[None, params.NUM_ACTIONS], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
        self.trainer = tf.train.AdamOptimizer()  # learning_rate=lrn_rate)

        self.updateModel = self.trainer.minimize(self.loss)

    def PrintQ(self, curr, params, env, sess):
        # print("hh", next, hh)
        visited = set()

        neighbours = env.GetNeighBours(curr, visited, params)
        a, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: neighbours})
        # print("curr=", curr, "a=", a, "allQ=", allQ, neighbours)
        print("curr=", curr, allQ, neighbours)

    def PrintAllQ(self, params, env, sess):
        print("         Q-values                          Next state")
        for curr in range(env.ns):
            self.PrintQ(curr, params, env, sess)


######################################################################################
# helpers
class Env:
    def __init__(self, siteMap):
        self.siteMap = siteMap

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

    def GetNeighBours(self, curr, visited, params):
        col = self.F[curr, :]
        ret = []
        for i in range(len(col)):
            if col[i] == 1 and i not in visited:
                ret.append(i)

        for i in range(len(ret), params.NUM_ACTIONS):
            ret.append(self.ns - 1)

        # random.shuffle(ret)

        # ret = np.empty([5,1])
        ret = np.array(ret)
        ret = ret.reshape([1, params.NUM_ACTIONS])
        # print("GetNeighBours", ret.shape, ret)

        return ret

    def Walk(self, start, params, sess, qn, printQ):
        visited = set()
        curr = start
        i = 0
        totReward = 0
        print(str(curr) + "->", end="")
        while True:
            # print("curr", curr)
            # print("hh", next, hh)
            neighbours = self.GetNeighBours(curr, visited, params)
            action, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.input: neighbours})
            action = action[0]
            next, reward, done = self.GetNextState(curr, action, neighbours)
            totReward += reward
            visited.add(next)

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

    def WalkAll(self, params, sess, qn):
        for start in range(self.ns):
            self.Walk(start, params, sess, qn, False)


######################################################################################

def Neural(epoch, curr, params, env, sess, qn, visited):
    # NEURAL
    # print("curr", curr, visited)
    neighbours = env.GetNeighBours(curr, visited, params)
    a, allQ = sess.run([qn.predict, qn.Qout], feed_dict={qn.input: neighbours})
    a = a[0]
    if np.random.rand(1) < params.eps:
        a = np.random.randint(0, params.NUM_ACTIONS)

    next, r, done = env.GetNextState(curr, a, neighbours)
    # print("curr=", curr, "a=", a, "next=", next, "r=", r, "allQ=", allQ)

    visited.add(next)

    # Obtain the Q' values by feeding the new state through our network
    if curr == env.ns - 1:
        targetQ = np.zeros([1, params.NUM_ACTIONS])
        maxQ1 = 0
    else:
        # print("  hh2", hh2)
        nextNeighbours = env.GetNeighBours(next, visited, params)
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


def UpdateQN(params, env, sess, qn, batch):
    batchSize = len(batch)
    neighbours = np.empty([batchSize, params.NUM_ACTIONS], dtype=np.int)
    targetQ = np.empty([batchSize, params.NUM_ACTIONS])

    i = 0
    for transition in batch:
        # print("transition", transition.neighbours.shape, transition.targetQ.shape)
        neighbours[i, :] = transition.neighbours
        targetQ[i, :] = transition.targetQ

        i += 1

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
    visited = set()

    while (True):
        transition = Neural(epoch, curr, params, env, sess, qn, visited)
        path.append(transition)
        curr = transition.next
        # print("visited", visited)

        if transition.done: break
    # print()

    return path


######################################################################################
class Corpus:
    def __init__(self, params):
        self.transitions = []

    def AddPath(self, path):
        for transition in path:
            self.transitions.append(transition)

    def GetBatch(self, maxBatchSize):
        batch = self.transitions[0:maxBatchSize]
        self.transitions = self.transitions[maxBatchSize:]

        return batch


######################################################################################

def Train(params, env, sess, qn):
    losses = []
    sumWeights = []
    corpus = Corpus(params)

    for epoch in range(params.max_epochs):
        startState = np.random.randint(0, env.ns)  # random start state
        path = Trajectory(epoch, startState, params, env, sess, qn)
        corpus.AddPath(path)

        while len(corpus.transitions) >= params.maxBatchSize:
            # print("corpusSize", corpusSize)

            batch = corpus.GetBatch(params.maxBatchSize)
            # print("batchSize", batchNeighbours.shape)
            # print("corpusNeighbours", corpusNeighbours.shape)
            # print("corpusTargetQ", corpusTargetQ.shape)

            loss, sumWeight = UpdateQN(params, env, sess, qn, batch)
            losses.append(loss)
            sumWeights.append(sumWeight)

        if epoch % params.walk == 0:
            print("\nepoch", epoch)
            qn.PrintAllQ(params, env, sess)
            env.WalkAll(params, sess, qn)
            # env.Walk(9, sess, qn, True)

        # add to batch
        stopState = path[-1].next
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
class MySQL:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        # paracrawl
        self.mydb = mysql.connector.connect(
            host=config["mysql"]["host"],
            user=config["mysql"]["user"],
            passwd=config["mysql"]["password"],
            database=config["mysql"]["database"],
            charset='utf8'
        )
        self.mydb.autocommit = False
        self.mycursor = self.mydb.cursor(buffered=True)


class Sitemap:
    def __init__(self, sqlconn, url):
        # all nodes with docs
        sql = "select url.id, url.document_id, document.lang, url.val from url, document where url.document_id = document.id and val like %s"
        val = (url + "%",)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchall()
        assert (res is not None)

        self.nodes = {}  # indexed by URL id
        self.nodesbyURL = {}  # indexed by URL
        for rec in res:
            # print("rec", rec[0], rec[1])
            id = len(self.nodes)
            node = Node(sqlconn, id, rec[0], rec[1], rec[2], rec[3])
            self.nodes[node.urlId] = node
            self.nodesbyURL[node.url] = node
        # print("nodes", len(self.nodes))

        self.nodesWithDoc = self.nodes.copy()
        print("nodesWithDoc", len(self.nodesWithDoc))

        # links between nodes, possibly to nodes without doc
        for node in self.nodesWithDoc.values():
            node.CreateLinks(sqlconn, self.nodes, self.nodesbyURL)
            # print("node", node.Debug())
        print("all nodes", len(self.nodes))

        # lang id
        self.langIds = {}

        # print out
        # for node in self.nodes.values():
        #    print("node", node.Debug())

        # node = Node(sqlconn, url, True)
        # print("node", node.docId, node.urlId)

    def GetLangId(self, langStr):
        if langStr in self.langIds:
            langId = self.langIds[langStr]
        else:
            langId = len(self.langIds)
            self.langIds[langStr] = langId
        return langId

    def GetRandomNode(self):
        l = list(self.nodesWithDoc.values())
        node = random.choice(l)
        return node

    def GetNode(self, url):
        node = self.nodesbyURL[url]
        return node


class Node:
    def __init__(self, sqlconn, id, urlId, docId, lang, url):
        self.id = id
        self.urlId = urlId
        self.docId = docId
        self.lang = lang
        self.url = url
        self.links = []
        self.aligned = False

        if self.docId is not None:
            sql = "select * from document_align where document1 = %s or document2 = %s"
            val = (self.docId, self.docId)
            # print("sql", sql)
            sqlconn.mycursor.execute(sql, val)
            res = sqlconn.mycursor.fetchall()
            # print("aligned",  self.url, self.docId, res)

            if len(res) > 0:
                self.aligned = True

        print(self.Debug())

    def Debug(self):
        return " ".join([str(self.id), str(self.urlId), StrNone(self.docId), StrNone(self.lang), str(len(self.links)),
                         str(self.aligned), self.url])

    def CreateLinks(self, sqlconn, nodes, nodesbyURL):
        # sql = "select id, text, url_id from link where document_id = %s"
        sql = "select link.id, link.text, link.text_lang, link.url_id, url.val from link, url where url.id = link.url_id and link.document_id = %s"
        val = (self.docId,)
        # print("sql", sql)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchall()
        assert (res is not None)

        for rec in res:
            text = rec[1]
            textLang = rec[2]
            urlId = rec[3]
            url = rec[4]
            # print("urlid", self.docId, text, urlId)

            if urlId in nodes:
                childNode = nodes[urlId]
                # print("child", self.docId, childNode.Debug())
            else:
                id = len(nodes)
                childNode = Node(sqlconn, id, urlId, None, None, url)
                nodes[childNode.urlId] = childNode
                nodesbyURL[childNode.url] = childNode

            Link = namedtuple("Link", "text textLang parentNode childNode")
            link = Link(text, textLang, self, childNode)
            self.links.append(link)

    def GetUnvisitedLinks(self, visited):
        links = []
        for link in self.links:
            childNode = link.childNode
            if childNode.docId is not None and childNode.urlId not in visited:
                links.append(link)
        return links


######################################################################################

######################################################################################


def Main():
    print("Starting")

    oparser = argparse.ArgumentParser(description="import-mysql")
    oparser.add_argument("--config-file", dest="configFile", required=True,
                         help="Path to config file (containing MySQL login etc.")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # =============================================================
    sqlconn = MySQL(options.configFile)
    # siteMap = Sitemap(sqlconn, "www.visitbritain.com")
    siteMap = Sitemap(sqlconn, "www.vade-retro.fr/")
    # =============================================================
    env = Env(siteMap)

    params = LearningParams()

    tf.reset_default_graph()
    qn = Qnetwork(params, env)
    init = tf.global_variables_initializer()
    print("qn.Qout", qn.Qout)

    with tf.Session() as sess:
        sess.run(init)

        losses, sumWeights = Train(params, env, sess, qn)
        print("Trained")

        qn.PrintAllQ(params, env, sess)
        env.WalkAll(params, sess, qn)

        plt.plot(losses)
        plt.show()

        plt.plot(sumWeights)
        plt.show()

    print("Finished")


if __name__ == "__main__":
    Main()
