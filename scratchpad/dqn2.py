#!/usr/bin/env python3

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
        self.alpha = 1.0  # 0.7
        self.max_epochs = 50001
        self.eps = 0.7
        self.maxBatchSize = 64
        self.minCorpusSize = 200
        self.trainNumIter = 10

        self.debug = False
        self.walk = 1000
        self.NUM_ACTIONS = 30


######################################################################################
class Qnetwork():
    def __init__(self, params, env):
        self.corpus = Corpus(params, self)

        # These lines establish the feed-forward part of the network used to choose actions
        EMBED_DIM = 3000

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
        unvisited = set()
        unvisited.add(0)

        childIds = env.GetChildIdsNP(curr, visited, unvisited, params)

        # action, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: childIds})
        action, allQ = self.Predict(sess, childIds)

        # print("   curr=", curr, "action=", action, "allQ=", allQ, childIds)
        print(curr, action, allQ, childIds)

    def PrintAllQ(self, params, env, sess):
        print("State         Q-values                          Next state")
        for curr in range(env.ns):
            self.PrintQ(curr, params, env, sess)

    def Predict(self, sess, input):
        action, allQ = sess.run([self.predict, self.Qout], feed_dict={self.input: input})
        action = action[0]

        return action, allQ

    def Update(self, sess, input, targetQ):
        _, loss, sumWeight = sess.run([self.updateModel, self.loss, self.sumWeight],
                                      feed_dict={self.input: input, self.nextQ: targetQ})
        return loss, sumWeight


######################################################################################
class Qnets():
    def __init__(self, params, env):
        self.q = []
        self.q.append(Qnetwork(params, env))
        self.q.append(Qnetwork(params, env))


######################################################################################
class Corpus:
    def __init__(self, params, qn):
        self.qn = qn
        self.transitions = []
        self.losses = []
        self.sumWeights = []

    def AddTransition(self, transition):
        self.transitions.append(transition)

    def AddPath(self, path):
        for transition in path:
            self.AddTransition(transition)

    def GetBatch(self, maxBatchSize):
        batch = self.transitions[0:maxBatchSize]
        self.transitions = self.transitions[maxBatchSize:]

        return batch

    def GetBatchWithoutDelete(self, maxBatchSize):
        batch = []

        size = len(self.transitions)
        for i in range(maxBatchSize):
            idx = np.random.randint(0, size)
            transition = self.transitions[idx]
            batch.append(transition)

        return batch

    def AddStopTransition(self, env, params):
        # stop state
        for i in range(10):
            targetQ = np.zeros([1, params.NUM_ACTIONS])
            childIds = env.GetStopChildIdsNP(params)
            transition = env.Transition(0, 0, True, np.array(childIds, copy=True), np.array(targetQ, copy=True))
            self.transitions.append(transition)

    def Train(self, sess, env, params):
        if len(self.transitions) >= params.minCorpusSize:
            self.AddStopTransition(env, params)

            for i in range(params.trainNumIter):
                batch = self.GetBatchWithoutDelete(params.maxBatchSize)
                loss, sumWeight = UpdateQN(params, env, sess, self.qn, batch)
                self.losses.append(loss)
                self.sumWeights.append(sumWeight)
            self.transitions.clear()


######################################################################################
# helpers
class MySQL:
    def __init__(self):
        # paracrawl
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="paracrawl_user",
            passwd="paracrawl_password",
            database="paracrawl",
            charset='utf8'
        )
        self.mydb.autocommit = False
        self.mycursor = self.mydb.cursor(buffered=True)


######################################################################################
class Env:
    def __init__(self, sqlconn, url):
        self.Transition = namedtuple("Transition", "curr next done childIds targetQ")

        self.numAligned = 0

        # all nodes with docs
        sql = "select url.id, url.document_id, document.lang, url.val from url, document where url.document_id = document.id and val like %s"
        val = (url + "%",)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchall()
        assert (res is not None)

        self.nodes = {}  # indexed by URL id
        self.nodesbyURL = {}  # indexed by URL
        self.nodesById = []

        # stop node
        node = Node(sqlconn, 0, 0, 0, "", "STOP")
        # self.nodes[node.urlId] = node
        # self.nodesbyURL[node.url] = node
        self.nodesById.append(node)

        for rec in res:
            # print("rec", rec[0], rec[1])
            id = len(self.nodesById)
            node = Node(sqlconn, id, rec[0], rec[1], rec[2], rec[3])
            self.nodes[node.urlId] = node
            self.nodesbyURL[node.url] = node
            self.nodesById.append(node)

            if node.aligned:
                self.numAligned += 1
        # print("nodes", len(self.nodes))
        print("numAligned", self.numAligned)

        # start node
        id = len(self.nodesById)
        rootNode = self.nodesbyURL[url]
        assert (rootNode is not None)
        startNode = Node(sqlconn, id, 0, 0, "", "START")
        startNode.CreateLink("", "", rootNode)
        # self.nodes[node.urlId] = startNode
        # self.nodesbyURL[node.url] = startNode
        self.nodesById.append(startNode)
        self.startNodeId = startNode.id
        # print("startNode", startNode.Debug())

        self.ns = len(self.nodesById)  # number of states

        self.nodesWithDoc = self.nodes.copy()
        print("nodesWithDoc", len(self.nodesWithDoc))

        # links between nodes, possibly to nodes without doc
        # for node in self.nodesWithDoc.values():
        for node in self.nodesById:
            node.CreateLinks(sqlconn, self.nodes, self.nodesbyURL, self.nodesById)
            print(node.Debug())

        print("all nodes", len(self.nodes))

        # lang id
        self.langIds = {}

        # print out
        # for node in self.nodes.values():
        #    print("node", node.Debug())

        # node = Node(sqlconn, url, True)
        # print("node", node.docId, node.urlId)

    def GetNextState(self, action, childIds):
        nextNodeId = childIds[0, action]
        nextNode = self.nodesById[nextNodeId]
        if nextNodeId == 0:
            rewardNode = 0
        elif nextNode.aligned:
            rewardNode = 8.5
        else:
            rewardNode = -1

        return nextNodeId, rewardNode

    def GetChildIdsNP(self, curr, visited, unvisited, params):
        currNode = self.nodesById[curr]
        # print("   currNode", curr, currNode.Debug())
        childIds = currNode.GetChildIds(visited, params)
        # print("   childIds", childIds)

        for childId in childIds:
            unvisited.add(childId)

        ret = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        i = 0
        for childId in unvisited:
            ret[0, i] = childId

            i += 1
            if i >= params.NUM_ACTIONS:
                print("overloaded", len(unvisited), unvisited)
                break

        return ret

    def GetStopChildIdsNP(self, params):
        childIds = np.zeros([1, params.NUM_ACTIONS])
        return childIds

    def Walk(self, start, params, sess, qn, printQ):
        numAligned = 0

        visited = set()
        unvisited = set()
        unvisited.add(0)

        curr = start
        i = 0
        totReward = 0
        mainStr = str(curr) + "->"
        debugStr = ""

        while True:
            # print("curr", curr)
            # print("hh", next, hh)
            childIds = self.GetChildIdsNP(curr, visited, unvisited, params)

            action, allQ = qn.Predict(sess, childIds)
            next, reward = self.GetNextState(action, childIds)
            totReward += reward
            visited.add(next)
            unvisited.remove(next)

            nextNode = self.nodesById[next]
            aligned = nextNode.aligned
            alignedStr = ""
            if aligned:
                alignedStr = "*"
                numAligned += 1

            if printQ:
                debugStr += "   " + str(curr) + "->" + str(next) + " " \
                            + str(action) + " " + str(allQ) + " " + str(childIds) + "\n"

            # print("(" + str(action) + ")", str(next) + "(" + str(reward) + ") -> ", end="")
            mainStr += str(next) + alignedStr + "->"
            curr = next

            if next == 0: break

            i += 1

        mainStr += " " + str(totReward)

        if printQ:
            print(debugStr, end="")
        print(mainStr)

        return numAligned

    def WalkAll(self, params, sess, qn):
        for start in range(self.ns):
            self.Walk(start, params, sess, qn, False)

    def GetNumberAligned(self, path):
        ret = 0
        for transition in path:
            next = transition.next
            nextNode = self.nodesById[next]
            if nextNode.aligned:
                ret += 1
        return ret


######################################################################################

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

        # print(self.Debug())

    def Debug(self):
        strLinks = ""
        for link in self.links:
            # strLinks += str(link.parentNode.id) + "->" + str(link.childNode.id) + " "
            strLinks += str(link.childNode.id) + " "

        return " ".join([str(self.id), str(self.urlId),
                         StrNone(self.docId), StrNone(self.lang),
                         str(self.aligned), self.url,
                         "links=", str(len(self.links)), ":", strLinks])

    def CreateLinks(self, sqlconn, nodes, nodesbyURL, nodesById):
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
                continue
                # id = len(nodes)
                # childNode = Node(sqlconn, id, urlId, None, None, url)
                # nodes[childNode.urlId] = childNode
                # nodesbyURL[childNode.url] = childNode
                # nodesById.append(childNode)

            self.CreateLink(text, textLang, childNode)

    def CreateLink(self, text, textLang, childNode):
        Link = namedtuple("Link", "text textLang parentNode childNode")
        link = Link(text, textLang, self, childNode)
        self.links.append(link)

    def GetChildIds(self, visited, params):
        childIds = []
        for link in self.links:
            childNode = link.childNode
            childNodeId = childNode.id
            # print("   ", childNode.Debug())
            if childNodeId != self.id and childNodeId not in visited:
                childIds.append(childNodeId)
        # print("   childIds", childIds)

        return childIds


######################################################################################

def UpdateQN(params, env, sess, qn, batch):
    batchSize = len(batch)
    # print("batchSize", batchSize)
    childIds = np.empty([batchSize, params.NUM_ACTIONS], dtype=np.int)
    targetQ = np.empty([batchSize, params.NUM_ACTIONS])

    i = 0
    for transition in batch:
        curr = transition.curr
        next = transition.next

        childIds[i, :] = transition.childIds
        targetQ[i, :] = transition.targetQ

        i += 1

    # _, loss, sumWeight = sess.run([qn.updateModel, qn.loss, qn.sumWeight], feed_dict={qn.input: childIds, qn.nextQ: targetQ})
    loss, sumWeight = qn.Update(sess, childIds, targetQ)

    # print("loss", loss)
    return loss, sumWeight


def Neural(epoch, curr, params, env, sess, qnA, qnB, visited, unvisited):
    assert (curr != 0)
    # print("curr", curr, visited, unvisited)
    childIds = env.GetChildIdsNP(curr, visited, unvisited, params)
    # print("   childIds", childIds, unvisited)

    action, allQ = qnA.Predict(sess, childIds)
    if np.random.rand(1) < params.eps:
        action = np.random.randint(0, params.NUM_ACTIONS)

    next, r = env.GetNextState(action, childIds)
    # print("   action", action, next)

    visited.add(next)
    unvisited.remove(next)
    nextUnvisited = unvisited.copy()

    if next == 0:
        done = True
        maxNextQ = 0.0
    else:
        assert (next != 0)
        done = False

        # Obtain the Q' values by feeding the new state through our network
        # print("  hh2", hh2)
        nextChildIds = env.GetChildIdsNP(next, visited, nextUnvisited, params)

        nextAction, nextQ = qnA.Predict(sess, nextChildIds)
        # print("  nextAction", nextAction, nextQ)

        assert (qnB == None)
        maxNextQ = np.max(nextQ)

        # _, nextQB = qnB.Predict(sess, nextChildIds)
        # maxNextQ = nextQB[0, nextAction]

    # targetQ = allQ
    targetQ = np.array(allQ, copy=True)
    # print("  targetQ", targetQ)
    newVal = r + params.gamma * maxNextQ
    targetQ[0, action] = (1 - params.alpha) * targetQ[0, action] + params.alpha * newVal
    # targetQ[0, action] = newVal
    # print("  targetQ", targetQ, maxNextQ)
    # print("  new Q", a, allQ)

    transition = env.Transition(curr, next, done, np.array(childIds, copy=True), np.array(targetQ, copy=True))
    return transition


def Trajectory(epoch, curr, params, env, sess, qns):
    visited = set()
    unvisited = set()
    unvisited.add(0)

    while (True):
        # tmp = np.random.rand(1)
        # if tmp > 0.5:
        #    qnA = qns.q[0]
        #    qnB = qns.q[1]
        # else:
        #    qnA = qns.q[1]
        #    qnB = qns.q[0]
        qnA = qns.q[0]
        qnB = None

        transition = Neural(epoch, curr, params, env, sess, qnA, qnB, visited, unvisited)

        qnA.corpus.AddTransition(transition)

        curr = transition.next
        # print("visited", visited)

        if transition.done: break
    # print("path", path)


def Train(params, env, sess, qns):
    for epoch in range(params.max_epochs):
        # print("epoch", epoch)
        # startState = np.random.randint(0, env.ns)  # random start state
        # startState = 30
        startState = env.startNodeId
        # print("startState", startState)

        Trajectory(epoch, startState, params, env, sess, qns)
        qns.q[0].corpus.Train(sess, env, params)
        qns.q[1].corpus.Train(sess, env, params)

        if epoch > 0 and epoch % params.walk == 0:
            qns.q[0].PrintAllQ(params, env, sess)
            print()
            numAligned = env.Walk(startState, params, sess, qns.q[0], True)
            print("epoch", epoch, "loss", qns.q[0].corpus.losses[-1], "eps", params.eps, "alpha", params.alpha)
            print()

            # numAligned = env.GetNumberAligned(path)
            # print("path", numAligned, env.numAligned)
            if numAligned >= env.numAligned - 5:
                # print("got them all!")
                # eps = 1. / ((i/50) + 10)
                params.eps *= .99
                params.eps = max(0.1, params.eps)

                params.alpha *= 0.99
                params.alpha = max(0.3, params.alpha)

    # LAST BATCH


######################################################################################

def Main():
    print("Starting")
    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    # =============================================================
    sqlconn = MySQL()
    # siteMap = Sitemap(sqlconn, "www.visitbritain.com")
    # =============================================================
    env = Env(sqlconn, "www.vade-retro.fr/")

    params = LearningParams()

    tf.reset_default_graph()
    qns = Qnets(params, env)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        qns.q[0].PrintAllQ(params, env, sess)
        # env.WalkAll(params, sess, qn)
        print()

        Train(params, env, sess, qns)
        print("Trained")

        # qn.PrintAllQ(params, env, sess)
        # env.WalkAll(params, sess, qn)

        startState = env.startNodeId
        env.Walk(startState, params, sess, qns.q[0], True)

        plt.plot(qns.q[0].corpus.losses)
        plt.plot(qns.q[1].corpus.losses)
        plt.show()

        plt.plot(qns.q[0].corpus.sumWeights)
        plt.plot(qns.q[1].corpus.sumWeights)
        plt.show()

    print("Finished")


if __name__ == "__main__":
    Main()
