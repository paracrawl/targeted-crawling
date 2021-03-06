#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
from matplotlib import pylab
import pylab as plt
import tensorflow as tf

from common import MySQL, Languages, Timer
from helpers import Env, Link


######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, saveDirPlots, deleteDuplicateTransitions, langPair):
        self.gamma = 0.999
        self.lrn_rate = 0.02
        self.alpha = 1.0 # 0.7
        self.max_epochs = 100001
        self.eps = 0.05
        self.maxBatchSize = 64
        self.minCorpusSize = 200
        self.trainNumIter = 30
        
        self.debug = False
        self.walk = 10
        self.NUM_ACTIONS = 30
        self.FEATURES_PER_ACTION = 1

        self.saveDir = saveDir
        self.saveDirPlots = saveDirPlots

        self.deleteDuplicateTransitions = deleteDuplicateTransitions

        self.discount = 0.995
        self.reward = 20.0
        self.cost = -1.0
        self.unusedActionCost = 0.0 #-555.0
        self.maxDocs = 9999999999

        self.mode = 'test'
        self.first_skip = 0
        self.last_skip = 0#100

        langPairList = langPair.split(",")
        assert(len(langPairList) == 2)
        self.langIds = [languages.GetLang(langPairList[0]), languages.GetLang(langPairList[1])] 
        #print("self.langs", self.langs)

######################################################################################
def NumParallelDocs(env, visited):
    ret = 0
    for urlId in visited:
        node = env.nodes[urlId]
        #print("node", node.Debug())

        if node.alignedNode is not None and node.alignedNode.urlId in visited:
            ret += 1

    return ret

######################################################################################
def dumb(env, maxDocs, params):
    ret = []
    todo = []
    todo.append(env.rootNode)

    visited = set()
    langsVisited = {}

    while len(todo) > 0 and len(visited) < maxDocs:
        node = todo.pop(0)
        #print("node", node.Debug())
        
        if node.urlId not in visited:
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)

            for link in node.links:
                childNode = link.childNode
                #print("   ", childNode.Debug())
                todo.append(childNode)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

    return ret

######################################################################################
def randomCrawl(env, maxDocs, params):
    ret = []
    todo = []
    todo.append(env.rootNode)

    visited = set()
    langsVisited = {}

    while len(todo) > 0 and len(visited) < maxDocs:
        idx = np.random.randint(0, len(todo))
        node = todo.pop(idx)
        #print("node", node.Debug())
        
        if node.urlId not in visited:
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)

            for link in node.links:
                childNode = link.childNode
                #print("   ", childNode.Debug())
                todo.append(childNode)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

    return ret

######################################################################################
def balanced(env, maxDocs, params):
    ret = []
    visited = set()
    langsVisited = {}
    langsTodo = {}

    startNode = env.nodes[sys.maxsize]
    #print("startNode", startNode.Debug())
    assert(len(startNode.links) == 1)
    link = next(iter(startNode.links))

    while link is not None and len(visited) < maxDocs:
        node = link.childNode
        if node.urlId not in visited:
            #print("node", node.Debug())
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)
    
            for link in node.links:
                #print("   ", childNode.Debug())
                AddTodo(langsTodo, visited, link)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

        link = PopLink(langsTodo, langsVisited, params)

    return ret

def PopLink(langsTodo, langsVisited, params):
    sum = 0
    # any nodes left to do
    for links in langsTodo.values():
        sum += len(links)
    if sum == 0:
        return None
    del sum

    # sum of all nodes visited
    sumAll = 0
    sumRequired = 0
    for lang, count in langsVisited.items():
        sumAll += count
        if lang in params.langIds:
            sumRequired += count
    sumRequired += 0.001 #1
    #print("langsVisited", sumAll, sumRequired, langsVisited)

    probs = {}
    for lang in params.langIds:
        if lang in langsVisited:
            count = langsVisited[lang]
        else:
            count = 0
        #print("langsTodo", lang, nodes)
        prob = 1.0 - float(count) / float(sumRequired)
        probs[lang] = prob
    #print("   probs", probs)

    links = None
    rnd = np.random.rand(1)
    #print("rnd", rnd, len(probs))
    cumm = 0.0
    for lang, prob in probs.items():
        cumm += prob
        #print("prob", prob, cumm)
        if cumm > rnd[0]:
            if lang in langsTodo:
                links = langsTodo[lang]
            break
    
    if links is not None and len(links) > 0:
        link = links.pop(0)
    else:
        link = RandomLink(langsTodo)
    #print("   node", node.Debug())
    return link

def RandomLink(langsTodo):
    while True:
        idx = np.random.randint(0, len(langsTodo))
        langs = list(langsTodo.keys())
        lang = langs[idx]
        links = langsTodo[lang]
        #print("idx", idx, len(nodes))
        if len(links) > 0:
            return links.pop(0)
    raise Exception("shouldn't be here")

def AddTodo(langsTodo, visited, link):
    childNode = link.childNode
    
    if childNode.urlId in visited:
        return

    parentNode = link.parentNode
    parentLang = parentNode.lang

    if parentLang not in langsTodo:
        langsTodo[parentLang] = []
    langsTodo[parentLang].append(link)

######################################################################################
######################################################################################
class Qnets():
    def __init__(self, params, max_env_maxLangId):
        self.q = []
        self.q.append(Qnetwork(params, max_env_maxLangId))
        self.q.append(Qnetwork(params, max_env_maxLangId))

######################################################################################
class Corpus:
    def __init__(self, params, qn):
        self.params = params
        self.qn = qn
        self.transitions = []
        self.losses = []
        self.sumWeights = []

    def AddTransition(self, transition):
        if self.params.deleteDuplicateTransitions:
            for currTrans in self.transitions:
                if currTrans.currURLId == transition.currURLId and currTrans.nextURLId == transition.nextURLId:
                    return
            # completely new trans
    
        self.transitions.append(transition)

    def GetBatchWithoutDelete(self, maxBatchSize):
        batch = []

        size = len(self.transitions)
        for i in range(maxBatchSize):
            idx = np.random.randint(0, size)
            transition = self.transitions[idx]
            batch.append(transition)

        return batch

    def Train(self, sess, env, params):
        if len(self.transitions) >= params.minCorpusSize:
            #for transition in self.transitions:
            #    print(DebugTransition(transition))

            for i in range(params.trainNumIter):
                batch = self.GetBatchWithoutDelete(params.maxBatchSize)
                loss, sumWeight = self.UpdateQN(params, env, sess, batch)
                self.losses.append(loss)
                self.sumWeights.append(sumWeight)
            self.transitions.clear()
        
    def UpdateQN(self, params, env, sess, batch):
        batchSize = len(batch)
        FeatureMatrices = np.empty([batchSize, self.qn.MAX_NODES, self.qn.NUM_FEATURES])
        targetQs = np.empty([batchSize, 1])


        i = 0
        for transition in batch:
            #curr = transition.curr
            #next = transition.next

            FeatureMatrices[i, :] = transition.featureMatrix
            targetQs[i, :] = transition.nextQValue

            i += 1

        #_, loss, sumWeight = sess.run([qn.updateModel, qn.loss, qn.sumWeight], feed_dict={qn.input: childIds, qn.nextQ: targetQ})
        TIMER.Start("UpdateQN.1")
        loss, sumWeight = self.qn.Update(sess, FeatureMatrix=FeatureMatrices, targetQ = targetQs)
        TIMER.Pause("UpdateQN.1")

        #print("loss", loss)
        return loss, sumWeight

######################################################################################
class Transition:
    def __init__(self, currURLId, nextURLId, FeatureMatrix, nextQValue):
        self.currURLId = currURLId
        self.nextURLId = nextURLId
        self.featureMatrix = FeatureMatrix
        self.nextQValue = nextQValue

    def DebugTransition(self):
        ret = str(self.currURLId) + "->" + str(self.nextURLId)
        return ret

######################################################################################
class Candidates:
    def __init__(self, params, env):
        self.params = params
        self.env = env
        # self.dict = {} # parent lang -> links[]
        self.coll = [] 

        #for langId in params.langIds:
        #    self.dict[langId] = []

    def copy(self):
        ret = Candidates(self.params, self.env)

        # for key, value in self.dict.items():
        #     #print("key", key, value)
        #     ret.dict[key] = value.copy()

        ret.coll = self.coll.copy()

        return ret
    
    def AddLink(self, link):
        # langId = link.parentNode.lang
        # if langId not in self.dict:
        #     self.dict[langId] = []
        # self.dict[langId].append(link)
        no_siblings = len(link.parentNode.links)
        self.coll.append((link,no_siblings))

    def AddLinks(self, node, visited, params):
        #print("   currNode", curr, currNode.Debug())
        newLinks = node.GetLinks(visited, params)

        for link in newLinks:
            self.AddLink(link)

    def Pop(self, action):
        # links = self.dict[action]
        # assert(len(links) > 0)

        # idx = np.random.randint(0, len(links))
        # link = links.pop(idx)

        # # remove all links going to same node
        # for otherLinks in self.dict.values():
        #     otherLinksCopy = otherLinks.copy()
        #     for otherLink in otherLinksCopy:
        #         if otherLink.childNode == link.childNode:
        #             otherLinks.remove(otherLink)
        assert(action < len(self.coll))
        link, _ = self.coll.pop(action)
        assert(link is not None)

        # remove all links going to same node
        collCopy = self.coll.copy()
        for otherLink, no_sib in collCopy:
            if otherLink.childNode == link.childNode:
                self.coll.remove((otherLink, no_sib))

        return link

    # def HasLinks(self, action):
    def HasLinks(self):
        # if action in self.dict and len(self.dict[action]) > 0:
        #     return True
        # else:
        #     return False

        return len(self.coll) > 0

    def Shuffle(self):
        np.random.shuffle(self.coll)

    # def Debug(self):
    #     # ret = ""
    #     # for lang in self.dict:
    #     #     ret += "lang=" + str(lang) + ":" + str(len(self.dict[lang])) + " "
    #         #links = self.dict[lang]
    #         #for link in links:
    #         #    ret += " " + link.parentNode.url + "->" + link.childNode.url
    #
    #     ret = ""
    #     ret += "lang=" + str(lang) + ":" + str(len(self.coll)) + " "
    #     return ret
    #     return ret
    
######################################################################################
def getChildrenDistribution(node, visited):

    # distribution idicies:
    # 0 -> visited & aligned
    # 1 -> visited & not-aligned
    # 2 -> not visited
    childrenDistribution = np.zeros(3)
    for link in node.links:
        if link.childNode in visited:
            if link.childNode.alignedNode and link.childNode.alignedNode in visited:
                childrenDistribution[0] += 1
            else:
                childrenDistribution[1] += 1
        else:
            childrenDistribution[2] += 1

    childrenDistribution = childrenDistribution / sum(childrenDistribution)

    return childrenDistribution

class Qnetwork():
    HIDDEN_DIM = 256
    NUM_FEATURES = 9
    MAX_NODES = 500

    def __init__(self, params, max_env_maxLangId):
        self.params = params
        self.corpus = Corpus(params, self)

        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.featureMatrix = tf.placeholder(shape=[None, self.MAX_NODES, self.NUM_FEATURES], dtype=tf.float32)
        self.inputVector = tf.reshape(self.featureMatrix, [-1, self.NUM_FEATURES])

        #self.inputVector = tf.layers.batch_normalization(self.inputVector, training=self.is_train)

        self.W1 = tf.Variable(tf.random_uniform([self.NUM_FEATURES, self.HIDDEN_DIM], 0, 0.1))
        self.b1 = tf.Variable(tf.random_uniform([1, self.HIDDEN_DIM], 0, 0.1))
        self.hidden1 = tf.matmul(self.inputVector, self.W1)
        self.hidden1 = tf.add(self.hidden1, self.b1)
        self.hidden1 = tf.nn.relu(self.hidden1)
        #print("self.hidden1", self.hidden1.shape)

        self.W2 = tf.Variable(tf.random_uniform([self.HIDDEN_DIM, self.HIDDEN_DIM], 0, 0.1))
        self.b2 = tf.Variable(tf.random_uniform([1, self.HIDDEN_DIM], 0, 0.1))
        self.hidden2 = tf.matmul(self.hidden1, self.W2)
        self.hidden2 = tf.add(self.hidden2, self.b2)
        self.hidden2 = tf.nn.relu(self.hidden2)
        #print("self.hidden2", self.hidden2.shape)

        self.W3 = tf.Variable(tf.random_uniform([self.HIDDEN_DIM, self.HIDDEN_DIM], 0, 0.1))
        self.b3 = tf.Variable(tf.random_uniform([1, self.HIDDEN_DIM], 0, 0.1))
        self.hidden3 = tf.matmul(self.hidden2, self.W3)
        self.hidden3 = tf.add(self.hidden3, self.b3)
        self.hidden3 = tf.nn.relu(self.hidden3)
        #print("self.hidden3", self.hidden3.shape)

        self.output = tf.reshape(self.hidden3, [-1, self.MAX_NODES, self.HIDDEN_DIM])
        self.qValues = tf.math.reduce_sum(self.output, axis=2, keep_dims=False)
        #print("self.qValue", self.qValue.shape)
       
        self.sumWeight = tf.reduce_sum(self.W1) \
                         + tf.reduce_sum(self.b1) \
                         + tf.reduce_sum(self.W2) \
                         + tf.reduce_sum(self.b2) \
                         + tf.reduce_sum(self.W3) \
                         + tf.reduce_sum(self.b3) 

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQValue = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.QValue = tf.math.reduce_max(self.qValues, axis=1, keep_dims=False)

        self.loss = tf.square(self.QValue - self.nextQValue)
        #self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
        self.trainer = tf.train.AdamOptimizer(learning_rate=params.lrn_rate) #learning_rate=lrn_rate)
        
        self.updateModel = self.trainer.minimize(self.loss)


    def PredictAll(self, env, params, sess, nodesVisited, langsVisited, cur_depth, num_crawled, avg_depth_crawled, candidates):
        qValues = {}
        maxQ = -9999999.0
        cache = {}


        totalLangsVisited = np.sum(langsVisited)
        totalNodes = len(env.nodes)
        FeatureMatrix = np.zeros([1, self.MAX_NODES, self.NUM_FEATURES])
        if len(candidates.coll) > 0:
            for idx in range(len(candidates.coll)):
                link, no_sib = candidates.coll[idx]
                langId = link.parentNode.lang

                if idx < self.MAX_NODES:
                    cur_depth = len(link.childNode.url.replace("://", "", 1).split("/"))
                    childDistribution = getChildrenDistribution(link.childNode, nodesVisited)

                    FeatureMatrix[0, idx, 0] = totalLangsVisited / totalNodes
                    FeatureMatrix[0, idx, 1] = langsVisited[0, params.langIds[0]] / totalLangsVisited
                    FeatureMatrix[0, idx, 2] = langsVisited[0, params.langIds[1]] / totalLangsVisited
                    FeatureMatrix[0, idx, 3] = avg_depth_crawled / 5
                    FeatureMatrix[0, idx, 4] = cur_depth / 5
                    FeatureMatrix[0, idx, 5] = no_sib / 10
                    FeatureMatrix[0, idx, 6] = childDistribution[0]
                    FeatureMatrix[0, idx, 7] = childDistribution[1]
                    FeatureMatrix[0, idx, 8] = childDistribution[2]

            qValues, QValue = sess.run([self.qValues, self.QValue],  feed_dict={self.featureMatrix: FeatureMatrix,
                                                                                self.is_train: False})

            qValues = qValues.ravel()[:len(candidates.coll)]
            argMax = np.argmax(qValues)

        else:
            qValues = [0.0]
            QValue = 0.0
            argMax = -1



        return qValues, QValue, argMax, FeatureMatrix

    def Update(self, sess, FeatureMatrix, targetQ):
        _, loss, sumWeight = sess.run([self.updateModel, self.loss, self.sumWeight],
                                      feed_dict={self.is_train: True,
                                                 self.featureMatrix: FeatureMatrix,
                                                 self.nextQValue: targetQ})

        return loss, sumWeight

######################################################################################
def GetNextState(env, params, action, visited, candidates):
    #print("candidates", action, candidates.Debug())
    # if action == 0:
    if action == -1:
        # no explicit stop state but no candidates
        stopNode = env.nodes[0]
        link = Link("", 0, stopNode, stopNode)
    else:
        # assert(candidates.HasLinks(action))
        assert(candidates.HasLinks())
        link = candidates.Pop(action)
 
    assert(link is not None)
    nextNode = link.childNode

    if nextNode.urlId == 0:
        reward = 0.0
    elif nextNode.alignedNode is not None and nextNode.alignedNode.urlId in visited:
        reward = params.reward * (params.discount ** len(visited))
        #print("   visited", visited)
        #print("   reward", reward)
        #print()
    else:
        #print("   non-rewarding")
        reward = params.cost

    return link, reward

def NeuralWalk(env, params, eps, candidates, visited, langsVisited, cur_depth, num_crawled, avg_depth_crawled, sess, qnA):
    qValues, maxQ, action, FeatureMatrix = \
        qnA.PredictAll(env, params, sess, visited, langsVisited, cur_depth, num_crawled, avg_depth_crawled, candidates)

    if action != -1 and np.random.rand(1) < eps:
        actions = range(len(qValues))
        action = np.random.choice(actions)
        maxQ = qValues[action]



    #print("action", action, qValues)

    #print("action", action, maxQ, qValues)
    link, reward = GetNextState(env, params, action, visited, candidates)
    assert(link is not None)
    #print("action", action, qValues, link.childNode.Debug(), reward)

    return qValues, maxQ, action, link, reward, FeatureMatrix

def Neural(env, params, candidates, visited, langsVisited, cur_depth, num_crawled, avg_depth_crawled, sess, qnA, qnB):
    _, maxQ, action, link, reward, FeatureMatrix = \
        NeuralWalk(env, params, params.eps, candidates, visited, langsVisited, cur_depth, num_crawled, avg_depth_crawled, sess, qnA)
    assert(link is not None)
    #print("action", action, qValues, link, reward)
    
    # calc nextMaxQ
    nextVisited = visited.copy()
    nextVisited.add(link.childNode.urlId)

    nextCandidates = candidates.copy()
    nextCandidates.AddLinks(link.childNode, nextVisited, params)
    no_sib = len(link.parentNode.links)
    nextLangsVisited = langsVisited.copy()
    nextLangsVisited[0, link.childNode.lang] += 1

    _, nextQValue, nextAction, _ = qnA.PredictAll(env, params, sess, nextVisited, nextLangsVisited, cur_depth, num_crawled, avg_depth_crawled, nextCandidates)

    newVal = reward + params.gamma * nextQValue
    targetQ = (1 - params.alpha) * maxQ + params.alpha * newVal

    transition = Transition(link.parentNode.urlId, 
                            link.childNode.urlId,
                            FeatureMatrix=FeatureMatrix,
                            nextQValue = targetQ)

    return transition

######################################################################################
def Trajectory(env, epoch, params, sess, qns):
    ret = []
    visited = set()
    langsVisited = np.zeros([1, env.maxLangId + 1]) # langId -> count
    candidates = Candidates(params, env)
    num_crawled = 0
    avg_depth_crawled = 0
    node = env.nodes[sys.maxsize]

    #stopNode = env.nodes[0]
    #link = Link("", 0, stopNode, stopNode)
    #candidates.AddLink(link)

    params.maxDocs = len(env.nodes) - params.last_skip

    while True:
        tmp = np.random.rand(1)
        if tmp > 0.5:
            qnA = qns.q[0]
            qnB = qns.q[1]
        else:
            qnA = qns.q[1]
            qnB = qns.q[0]

        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        langsVisited[0, node.lang] += 1
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)
        
        cur_depth = len(node.url.replace("://", "", 1).split("/"))
        avg_depth_crawled = (avg_depth_crawled * num_crawled + cur_depth) / (num_crawled + 1)
        num_crawled += 1

        transition = Neural(env, params, candidates, visited, langsVisited, cur_depth, num_crawled, avg_depth_crawled, sess, qnA, qnB)
        candidates.Shuffle()
        if transition.nextURLId == 0:
            break
        else:
            qnA.corpus.AddTransition(transition)
            node = env.nodes[transition.nextURLId]

        if len(visited) > params.maxDocs:
            break

    return ret

######################################################################################
def Walk(env, params, sess, qns):
    ret = []
    visited = set()
    langsVisited = np.zeros([1, env.maxLangId + 1]) # langId -> count
    candidates = Candidates(params, env)
    num_crawled = 0
    avg_depth_crawled = 0
    node = env.nodes[sys.maxsize]

    #stopNode = env.nodes[0]
    #link = Link("", 0, stopNode, stopNode)
    #candidates.AddLink(link)

    mainStr = "nodes:" + str(node.urlId)
    rewardStr = "rewards:"

    i = 0
    numAligned = 0
    totReward = 0.0
    totDiscountedReward = 0.0
    discount = 1.0

    params.maxDocs = 9999999999

    while True:
        qnA = qns.q[0]
        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        langsVisited[0, node.lang] += 1
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        cur_depth = len(node.url.replace("://", "", 1).split("/"))
        avg_depth_crawled = (avg_depth_crawled * num_crawled + cur_depth) / (num_crawled + 1)
        num_crawled += 1

        #print("candidates", candidates.Debug())
        qValues, _, action, link, reward, _ = \
            NeuralWalk(env, params, 0.0, candidates, visited, langsVisited, cur_depth, num_crawled, avg_depth_crawled, sess, qnA)
        node = link.childNode
        print("action", action)

        totReward += reward
        totDiscountedReward += discount * reward

        mainStr += "->" + str(node.urlId)
        rewardStr += "->" + str(reward)

        if node.alignedNode is not None:
            mainStr += "*"
            numAligned += 1

        discount *= params.gamma
        i += 1

        candidates.Shuffle()

        if node.urlId == 0:
            break

        if len(visited) > params.maxDocs:
            break

    mainStr += " " + str(i) 
    rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)

    print(mainStr)
    print(rewardStr)
    return ret

######################################################################################
def Train(params, sess, saver, env, qns, env_test):
    totRewards = []
    totDiscountedRewards = []

    for epoch in range(params.max_epochs):
        print("epoch", epoch)
        TIMER.Start("Trajectory")
        _ = Trajectory(env, epoch, params, sess, qns)

        TIMER.Pause("Trajectory")

        TIMER.Start("Update")
        qns.q[0].corpus.Train(sess, env, params)
        qns.q[1].corpus.Train(sess, env, params)
        TIMER.Pause("Update")

        if epoch > 0 and epoch % params.walk == 0:
            arrDumb = dumb(env, len(env.nodes), params)
            arrRandom = randomCrawl(env, len(env.nodes), params)
            arrBalanced = balanced(env, len(env.nodes), params)
            arrRL = Walk(env, params, sess, qns)

            arrDumb_test = dumb(env_test, len(env_test.nodes), params)
            arrRandom_test = randomCrawl(env_test, len(env_test.nodes), params)
            arrBalanced_test = balanced(env_test, len(env_test.nodes), params)
            arrRL_test = Walk(env_test, params, sess, qns)

            print("epoch", epoch)

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(arrDumb, label="dumb_train", color='maroon')
            ax.plot(arrRandom, label="random_train", color='firebrick')
            ax.plot(arrBalanced, label="balanced_train", color='red')
            ax.plot(arrRL, label="RL_train", color='salmon')

            ax.legend(loc='upper left')
            plt.xlabel('#crawled')
            plt.ylabel('#found')

            fig.savefig("{}/{}_epoch{}.png".format(params.saveDirPlots, 'Train', epoch))
            fig.show()

            plt.pause(0.001)

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(arrDumb_test, label="dumb_test", color='navy')
            ax.plot(arrRandom_test, label="random_test", color='blue')
            ax.plot(arrBalanced_test, label="balanced_test", color='dodgerblue')
            ax.plot(arrRL_test, label="RL_test", color='lightskyblue')

            ax.legend(loc='upper left')
            plt.xlabel('#crawled')
            plt.ylabel('#found')
            fig.savefig("{}/{}_epoch{}".format(params.saveDirPlots, 'Test', epoch))
            fig.show()

            plt.pause(0.001)

    return totRewards, totDiscountedRewards

######################################################################################
def main():
    global TIMER
    TIMER = Timer()

    oparser = argparse.ArgumentParser(description="intelligent crawling with q-learning")
    oparser.add_argument("--config-file", dest="configFile", required=True,
                         help="Path to config file (containing MySQL login etc.)")
    oparser.add_argument("--language-pair", dest="langPair", required=True,
                         help="The 2 language we're interested in, separated by ,")
    oparser.add_argument("--save-dir", dest="saveDir", default=".",
                         help="Directory that model WIP are saved to. If existing model exists then load it")
    oparser.add_argument("--save-plots", dest="saveDirPlots", default="plot",
                     help="Directory ")
    oparser.add_argument("--delete-duplicate-transitions", dest="deleteDuplicateTransitions",
                         default=False, help="If True then only unique transition are used in each batch")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    sqlconn = MySQL(options.configFile)

    languages = Languages(sqlconn.mycursor)
    params = LearningParams(languages, options.saveDir, options.saveDirPlots, options.deleteDuplicateTransitions, options.langPair)


    #hostName = "http://www.visitbritain.com/"
    #
    hostName = "http://www.buchmann.ch/"
    #hostName = "http://vade-retro.fr/"    # smallest domain for debugging


    hostName_test = "http://www.visitbritain.com/"
    #hostName_test = "http://www.buchmann.ch/"
    #hostName_test = "http://vade-retro.fr/"    # smallest domain for debugging

    env = Env(sqlconn, hostName)
    env_test = Env(sqlconn, hostName_test)

    # change language of start node. 0 = stop
    env.nodes[sys.maxsize].lang = languages.GetLang("None")
    env_test.nodes[sys.maxsize].lang = languages.GetLang("None")
    #for node in env.nodes.values():
    #    print(node.Debug())

    max_env_maxLangId = max([env.maxLangId, env_test.maxLangId])
    env.maxLangId = env_test.maxLangId = max_env_maxLangId

    tf.reset_default_graph()
    qns = Qnets(params, max_env_maxLangId)
    init = tf.global_variables_initializer()

    saver = None #tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        totRewards, totDiscountedRewards = Train(params, sess, saver, env, qns, env_test)

        #params.debug = True
        arrDumb = dumb(env, len(env.nodes), params)
        arrRandom = randomCrawl(env, len(env.nodes), params)
        arrBalanced = balanced(env, len(env.nodes), params)
        arrRL = Walk(env, params, sess, qns)
        #print("arrDumb", arrDumb)
        #print("arrBalanced", arrBalanced)
        
        plt.plot(arrDumb, label="dumb")
        plt.plot(arrRandom, label="random")
        plt.plot(arrBalanced, label="balanced")
        plt.plot(arrRL, label="RL")
        plt.legend(loc='upper left')
        plt.xlabel('#crawled')
        plt.ylabel('#found')
        plt.show()

######################################################################################
main()
