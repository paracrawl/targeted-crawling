#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import tensorflow as tf
from tldextract import extract

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pylab as plt

from common import GetLanguages, Languages, Timer
from helpers import GetEnvs, GetVistedSiblings, GetMatchedSiblings, Env, Link

######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, saveDirPlots, deleteDuplicateTransitions, langPair, maxLangId, defaultLang):
        self.gamma = 0.999
        self.lrn_rate = 0.001
        self.alpha = 0.7
        self.max_epochs = 100001
        self.eps = 0.1
        self.maxBatchSize = 1
        self.minCorpusSize = 200
        self.overSampling = 1
        
        self.debug = False
        self.walk = 10
        self.NUM_ACTIONS = 30
        self.FEATURES_PER_ACTION = 1

        self.saveDir = saveDir
        self.saveDirPlots = saveDirPlots

        self.deleteDuplicateTransitions = deleteDuplicateTransitions
        
        self.reward = 100.0 #17.0
        self.cost = -1.0
        self.unusedActionCost = 0.0 #-555.0
        self.maxDocs = 9999999999

        self.maxLangId = maxLangId
        self.defaultLang = defaultLang
        self.MAX_NODES = 1000

        langPairList = langPair.split(",")
        assert(len(langPairList) == 2)
        self.langIds = np.empty([1,2], dtype=np.int32)
        self.langIds[0,0] = languages.GetLang(langPairList[0])
        self.langIds[0,1] = languages.GetLang(langPairList[1])
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
    for i in range(params.langIds.shape[1]):
        lang = params.langIds[0, i]
        #print("lang", lang)
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
def SavePlots(sess, qns, params, envs, saveDirPlots, epoch, sset):
    for env in envs:
        SavePlot(sess, qns, params, env, saveDirPlots, epoch, sset)

######################################################################################
def SavePlot(sess, qns, params, env, saveDirPlots, epoch, sset):
    print("Walking", env.rootURL)
    arrDumb = dumb(env, len(env.nodes), params)
    arrRandom = randomCrawl(env, len(env.nodes), params)
    arrBalanced = balanced(env, len(env.nodes), params)
    arrRL, totReward, totDiscountedReward = Walk(env, params, sess, qns)

    url = env.rootURL
    domain = extract(url).domain

    warmUp = 200
    avgRandom = avgBalanced = avgRL = 0.0
    for i in range(len(arrDumb)):
        if i > warmUp and arrDumb[i] > 0:
            avgRandom += arrRandom[i] / arrDumb[i]
            avgBalanced += arrBalanced[i] / arrDumb[i]
            avgRL += arrRL[i] / arrDumb[i]

    avgRandom = avgRandom / (len(arrDumb) - warmUp)
    avgBalanced = avgBalanced / (len(arrDumb) - warmUp)
    avgRL = avgRL / (len(arrDumb) - warmUp)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(arrDumb, label="dumb ", color='maroon')
    ax.plot(arrRandom, label="random {0:.1f}".format(avgRandom), color='firebrick')
    ax.plot(arrBalanced, label="balanced {0:.1f}".format(avgBalanced), color='red')
    ax.plot(arrRL, label="RL {0:.1f} {1:.1f}".format(avgRL, totDiscountedReward), color='salmon')

    ax.legend(loc='upper left')
    plt.xlabel('#crawled')
    plt.ylabel('#found')
    plt.title("{sset} {domain}".format(sset=sset, domain=domain))

    fig.savefig("{dir}/{sset}-{domain}-{epoch}.png".format(dir=saveDirPlots, sset=sset, domain=domain, epoch=epoch))
    fig.show()

######################################################################################
class Qnets():
    def __init__(self, params):
        self.q = []
        self.q.append(Qnetwork(params))
        self.q.append(Qnetwork(params))

######################################################################################
class Corpus:
    def __init__(self, params, qn):
        self.params = params
        self.qn = qn
        self.transitions = []
        self.losses = []

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

    def Train(self, sess, params):
        if len(self.transitions) >= params.minCorpusSize:
            #for transition in self.transitions:
            #    print(DebugTransition(transition))

            numIter = len(self.transitions) * params.overSampling / params.maxBatchSize
            numIter = int(numIter) + 1
            #print("numIter", numIter, len(self.transitions), params.overSampling, params.maxBatchSize)
            for i in range(numIter):
                batch = self.GetBatchWithoutDelete(params.maxBatchSize)
                loss = self.UpdateQN(params, sess, batch)
                self.losses.append(loss)
            self.transitions.clear()
        
    def UpdateQN(self, params, sess, batch):
        batchSize = len(batch)
        #print("batchSize", batchSize)
        numActions = np.empty([batchSize, 1], dtype=np.int)
        linkLang = np.empty([batchSize, self.params.MAX_NODES], dtype=np.int)
        mask = np.empty([batchSize, self.params.MAX_NODES], dtype=np.bool)
        numSiblings = np.empty([batchSize, self.params.MAX_NODES], dtype=np.float32)
        numVisitedSiblings = np.empty([batchSize, self.params.MAX_NODES], dtype=np.float32)
        numMatchedSiblings = np.empty([batchSize, self.params.MAX_NODES], dtype=np.float32)
        langIds = np.empty([batchSize, 2], dtype=np.int)
        langsVisited = np.empty([batchSize, params.maxLangId + 1])
        targetQ = np.empty([batchSize, self.params.MAX_NODES])

        i = 0
        for transition in batch:
            #curr = transition.curr
            #next = transition.next
            assert(transition.numActions == transition.targetQ.shape[1])
            numActions[i, 0] = transition.numActions
            linkLang[i, :] = transition.linkLang
            mask[i, :] = transition.mask
            numSiblings[i, :] = transition.numSiblings
            numVisitedSiblings[i, :] = transition.numVisitedSiblings
            numMatchedSiblings[i, :] = transition.numMatchedSiblings
            langIds[i, :] = transition.langIds
            langsVisited[i, :] = transition.langsVisited
            targetQ[i, 0:transition.numActions] = transition.targetQ

            i += 1

        TIMER.Start("UpdateQN.1")
        loss = self.qn.Update(sess, numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, langIds, langsVisited, targetQ)
        TIMER.Pause("UpdateQN.1")

        #print("loss", loss)
        return loss

######################################################################################
class Transition:
    def __init__(self, currURLId, nextURLId, numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, langIds, langsVisited, targetQ):
        self.currURLId = currURLId
        self.nextURLId = nextURLId 
        self.numActions = numActions
        self.linkLang = np.array(linkLang, copy=True) 
        self.mask = np.array(mask, copy=True) 
        self.numSiblings = np.array(numSiblings, copy=True) 
        self.numVisitedSiblings = np.array(numVisitedSiblings, copy=True) 
        self.numMatchedSiblings = np.array(numMatchedSiblings, copy=True) 
        self.langIds = langIds 
        self.langsVisited = np.array(langsVisited, copy=True)
        self.targetQ = np.array(targetQ, copy=True)

    def DebugTransition(self):
        ret = str(self.currURLId) + "->" + str(self.nextURLId)
        return ret

######################################################################################
class Candidates:
    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.dict = {} # key -> links[]

        #for langId in params.langIds:
        #    self.dict[langId] = []

    def copy(self):
        ret = Candidates(self.params, self.env)

        for key, value in self.dict.items():
            #print("key", key, value)
            ret.dict[key] = value.copy()

        return ret
    
    def AddLink(self, link, visited):
        langId = link.parentNode.lang
        numSiblings = len(link.parentNode.links)
        
        numVisitedSiblings = GetVistedSiblings(link.childNode.urlId, link.parentNode, visited)
        numVisitedSiblings = len(numVisitedSiblings)

        matchedSiblings = GetMatchedSiblings(link.childNode.urlId, link.parentNode, visited)
        numMatchedSiblings = len(matchedSiblings)
        
        #print("numSiblings", numSiblings, numMatchedSiblings, link.childNode.url)
        #for sibling in link.parentNode.links:
        #    print("   sibling", sibling.childNode.url)

        key = (langId,numSiblings, numVisitedSiblings, numMatchedSiblings) 
        if key not in self.dict:
            self.dict[key] = []
        self.dict[key].append(link)
        
    def AddLinks(self, node, visited, params):
        #print("   currNode", curr, currNode.Debug())
        newLinks = node.GetLinks(visited, params)

        for link in newLinks:
            self.AddLink(link, visited)

    def Pop(self, key):
        links = self.dict[key]
        assert(len(links) > 0)

        idx = np.random.randint(0, len(links))
        link = links.pop(idx)

        # remove all links going to same node
        for otherLinks in self.dict.values():
            otherLinksCopy = otherLinks.copy()
            for otherLink in otherLinksCopy:
                if otherLink.childNode == link.childNode:
                    otherLinks.remove(otherLink)

        return link

    def Count(self):
        ret = 0
        for _, dict in self.dict.items():
            ret += len(dict)
        return ret

    def GetFeatures(self):
        numActions = 0
        linkLang = np.zeros([1, self.params.MAX_NODES], dtype=np.int32)
        numSiblings = np.zeros([1, self.params.MAX_NODES], dtype=np.int32)
        numVisitedSiblings = np.zeros([1, self.params.MAX_NODES], dtype=np.int32)
        numMatchedSiblings = np.zeros([1, self.params.MAX_NODES], dtype=np.int32)

        mask = np.full([1, self.params.MAX_NODES], False, dtype=np.bool)
        
        for key, nodes in self.dict.items():
            if len(nodes) > 0:
                assert(numActions < self.params.MAX_NODES)
                linkLang[0, numActions] = key[0]
                numSiblings[0, numActions] = key[1]
                numVisitedSiblings[0, numActions] = key[2]
                numMatchedSiblings[0, numActions] = key[3]

                mask[0, numActions] = True
                numActions += 1

        return numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings

    def Debug(self):
        ret = ""
        for lang in self.dict:
            ret += "lang=" + str(lang) + ":" + str(len(self.dict[lang])) + " "
            #links = self.dict[lang]
            #for link in links:
            #    ret += " " + link.parentNode.url + "->" + link.childNode.url
        return ret
    
######################################################################################
class Qnetwork():
    def __init__(self, params):
        self.params = params
        self.corpus = Corpus(params, self)

        HIDDEN_DIM = 512
        NUM_FEATURES = params.maxLangId + 1

        # mask
        self.mask = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.bool)

        # graph represention
        self.langIds = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.langsVisited = tf.placeholder(shape=[None, NUM_FEATURES], dtype=tf.float32)
        self.numActions = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # link representation
        self.linkLang = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numVisitedSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numMatchedSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)

        # batch size
        self.batchSize = tf.shape(self.linkLang)[0]
        
        # network
        self.input = tf.concat([self.langIds, self.langsVisited, self.numActions], 1)
        #print("self.input", self.input.shape)

        self.W1 = tf.Variable(tf.random_uniform([NUM_FEATURES + 3, HIDDEN_DIM], 0, 0.01))
        self.b1 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden1 = tf.matmul(self.input, self.W1)
        self.hidden1 = tf.add(self.hidden1, self.b1)
        self.hidden1 = tf.nn.relu(self.hidden1)
        #print("self.hidden1", self.hidden1.shape)

        self.W2 = tf.Variable(tf.random_uniform([HIDDEN_DIM, HIDDEN_DIM], 0, 0.01))
        self.b2 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden2 = tf.matmul(self.hidden1, self.W2)
        self.hidden2 = tf.add(self.hidden2, self.b2)
        self.hidden2 = tf.nn.relu(self.hidden2)
        #print("self.hidden2", self.hidden2.shape)

        self.W3 = tf.Variable(tf.random_uniform([HIDDEN_DIM, HIDDEN_DIM], 0, 0.01))
        self.b3 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden3 = tf.matmul(self.hidden2, self.W3)
        self.hidden3 = tf.add(self.hidden3, self.b3)
        self.hidden3 = tf.nn.relu(self.hidden3)
        #print("self.hidden3", self.hidden3.shape)

        # link-specific
        self.hidden3 = tf.transpose(self.hidden3)

        #self.linkLangEmbedding = tf.nn.embedding_lookup(self.embeddings, self.linkLang)
        #self.linkLangEmbedding = tf.reshape(self.linkLangEmbedding, [self.batchSize * self.params.MAX_NODES, HIDDEN_DIM])
        #print("self.linkLang", self.linkLang.shape, self.linkLangEmbedding)

        self.WlinkSpecific = tf.Variable(tf.random_uniform([4, HIDDEN_DIM], 0, 0.01))
        self.blinkSpecific = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))

        self.linkSpecific = tf.concat([self.linkLang, self.numSiblings, self.numVisitedSiblings, self.numMatchedSiblings], 0)
        self.linkSpecific = tf.transpose(self.linkSpecific)
        self.linkSpecific = tf.matmul(self.linkSpecific, self.WlinkSpecific)
        self.linkSpecific = tf.add(self.linkSpecific, self.blinkSpecific)
        self.linkSpecific = tf.nn.relu(self.linkSpecific)

        # final q-values
        self.hidden3 = tf.matmul(self.linkSpecific, self.hidden3)
        self.hidden3 = tf.transpose(self.hidden3)

        self.qValues = tf.boolean_mask(self.hidden3, self.mask, axis=0)

        #self.hidden3 = tf.math.reduce_sum(self.hidden3, axis=1)
        #self.qValues = self.hidden3
        #print("self.qValues", self.qValue.shapes)
       
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        nextQMasked = tf.boolean_mask(self.nextQ, self.mask, axis=0)

        self.loss = nextQMasked - self.qValues
        self.loss = tf.reduce_max(tf.square(self.loss))
        #self.loss = tf.reduce_mean(tf.square(self.loss))
        #self.loss = tf.reduce_sum(tf.square(self.loss))
        
        #self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
        self.trainer = tf.train.AdamOptimizer(learning_rate=params.lrn_rate)
        
        self.updateModel = self.trainer.minimize(self.loss)

        #self.sumWeight = tf.reduce_sum(self.W1) \
        #                 + tf.reduce_sum(self.b1) \
        #                 + tf.reduce_sum(self.W2) \
        #                 + tf.reduce_sum(self.b2) \
        #                 + tf.reduce_sum(self.W3) \
        #                 + tf.reduce_sum(self.b3) 

    def PredictAll(self, env, sess, langIds, langsVisited, candidates):
        numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings = candidates.GetFeatures()
        
        numActionsNP = np.empty([1,1], dtype=np.int32)
        numActionsNP[0,0] = numActions

        if numActions > 0:
            #print("linkLang", numActions, linkLang.shape)
            #print("mask", mask.shape, mask)
            
            (qValues, ) = sess.run([self.qValues, ], 
                                    feed_dict={self.linkLang: linkLang,
                                        self.numActions: numActionsNP,
                                        self.mask: mask,
                                        self.numSiblings: numSiblings,
                                        self.numVisitedSiblings: numVisitedSiblings,
                                        self.numMatchedSiblings: numMatchedSiblings,
                                        self.langIds: langIds,
                                        self.langsVisited: langsVisited})
            #qValues = qValues[0]
            #print("hidden3", hidden3.shape, hidden3)
            #print("qValues", qValues.shape, qValues)
            #print("linkSpecific", linkSpecific.shape)
            #print("numSiblings", numSiblings.shape)
            #print("numVisitedSiblings", numVisitedSiblings.shape)
            #print("numMatchedSiblings", numMatchedSiblings.shape)
            qValues = np.reshape(qValues, [1, qValues.shape[0] ])
            #print("   qValues", qValues)
            #print()

            action = np.argmax(qValues[0, :numActions])
            maxQ = qValues[0, action]
            #print("newAction", action, maxQ)
        else:
            maxQ = 0.0 #-99999.0
            action = -1
            qValues = np.zeros([1, self.params.MAX_NODES], dtype=np.float32)


        #print("qValues", qValues.shape, qValues, action, maxQ)
        return numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, qValues, maxQ, action

    def Update(self, sess, numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, langIds, langsVisited, targetQ):
        #print("input", linkLang.shape, langIds.shape, langFeatures.shape, targetQ.shape)
        #print("targetQ", targetQ)
        _, loss = sess.run([self.updateModel, self.loss], 
                                    feed_dict={self.linkLang: linkLang, 
                                            self.numActions: numActions,
                                            self.mask: mask,
                                            self.numSiblings: numSiblings,
                                            self.numVisitedSiblings: numVisitedSiblings,
                                            self.numMatchedSiblings: numMatchedSiblings,
                                            self.langIds: langIds, 
                                            self.langsVisited: langsVisited,
                                            self.nextQ: targetQ})
        #print("loss", loss, numActions)
        return loss

######################################################################################
def GetNextState(env, params, action, visited, candidates, linkLang, numSiblings, numVisitedSiblings, numMatchedSiblings):
    #print("candidates", action, candidates.Debug())
    if action == -1:
        # no explicit stop state but no candidates
        stopNode = env.nodes[0]
        link = Link("", 0, stopNode, stopNode)
    else:
        langId = linkLang[0, action]
        numSiblings1 = numSiblings[0, action]
        numVisitedSiblings1 = numVisitedSiblings[0, action]
        numMatchedSiblings1 = numMatchedSiblings[0, action]
        key = (langId, numSiblings1, numVisitedSiblings1, numMatchedSiblings1)
        link = candidates.Pop(key)
 
    assert(link is not None)
    nextNode = link.childNode
    #print("   nextNode", nextNode.Debug())

    if nextNode.urlId == 0:
        #print("   stop")
        reward = 0.0
    elif nextNode.alignedNode is not None and nextNode.alignedNode.urlId in visited:
        reward = params.reward
        #print("   visited", visited)
        #print("   reward", reward)
        #print()
    else:
        #print("   non-rewarding")
        reward = params.cost

    return link, reward

######################################################################################
def NeuralWalk(env, params, eps, candidates, visited, langsVisited, sess, qnA):
    numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, qValues, maxQ, action = qnA.PredictAll(env, sess, params.langIds, langsVisited, candidates)
    #print("action", action, linkLang, qValues)
    if action >= 0:
        if np.random.rand(1) < eps:
            #print("actions", type(actions), actions)
            action = np.random.randint(0, numActions)
            maxQ = qValues[0, action]
            #print("random")
        #print("action", action, qValues)

    #print("action", action, maxQ, qValues)
    link, reward = GetNextState(env, params, action, visited, candidates, linkLang, numSiblings, numVisitedSiblings, numMatchedSiblings)
    assert(link is not None)
    #print("action", action, qValues, link.childNode.Debug(), reward)

    return numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, qValues, maxQ, action, link, reward

######################################################################################
def Neural(env, params, candidates, visited, langsVisited, sess, qnA, qnB):
    numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, qValues, maxQ, action, link, reward = NeuralWalk(env, params, params.eps, candidates, visited, langsVisited, sess, qnA)
    assert(link is not None)
    
    # calc nextMaxQ
    nextVisited = visited.copy()
    nextVisited.add(link.childNode.urlId)

    nextCandidates = candidates.copy()
    nextCandidates.AddLinks(link.childNode, nextVisited, params)

    nextLangsVisited = langsVisited.copy()
    nextLangsVisited[0, link.childNode.lang] += 1

    if nextCandidates.Count() > 0:
        _, _, _, _, _, _, _, _, nextAction = qnA.PredictAll(env, sess, params.langIds, nextLangsVisited, nextCandidates)
        #print("nextAction", nextAction, nextLangRequested, nextCandidates.Debug())
        _, _, _, _, _, _, nextQValuesB, _, _ = qnB.PredictAll(env, sess, params.langIds, nextLangsVisited, nextCandidates)
        nextMaxQ = nextQValuesB[0, nextAction]
        #print("nextMaxQ", nextMaxQ, nextMaxQB, nextQValuesA[0, nextAction])
    else:
        nextMaxQ = 0

    newVal = reward + params.gamma * nextMaxQ
    targetQ = (1 - params.alpha) * maxQ + params.alpha * newVal
    qValues[0, action] = targetQ

    transition = Transition(link.parentNode.urlId, 
                            link.childNode.urlId,
                            numActions,
                            linkLang,
                            mask,
                            numSiblings,
                            numVisitedSiblings,
                            numMatchedSiblings,
                            params.langIds,
                            langsVisited,
                            qValues)

    return transition

######################################################################################
def Trajectory(env, epoch, params, sess, qns):
    ret = []
    visited = set()
    langsVisited = np.zeros([1, params.maxLangId + 1]) # langId -> count
    candidates = Candidates(params, env)
    node = env.nodes[sys.maxsize]

    #stopNode = env.nodes[0]
    #link = Link("", 0, stopNode, stopNode)
    #candidates.AddLink(link)

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

        transition = Neural(env, params, candidates, visited, langsVisited, sess, qnA, qnB)

        if transition.nextURLId == 0:
            break
        else:
            tmp = np.random.rand(1)
            if tmp > 0.5:
                corpus = qnA.corpus
            else:
                corpus = qnB.corpus

            corpus.AddTransition(transition)
            node = env.nodes[transition.nextURLId]

        if len(visited) > params.maxDocs:
            break

    return ret

######################################################################################
def Walk(env, params, sess, qns):
    ret = []
    visited = set()
    langsVisited = np.zeros([1, params.maxLangId + 1]) # langId -> count
    candidates = Candidates(params, env)
    node = env.nodes[sys.maxsize]

    #stopNode = env.nodes[0]
    #link = Link("", 0, stopNode, stopNode)
    #candidates.AddLink(link)

    mainStr = "lang:" + str(node.lang)
    rewardStr = "rewards:"
    actionStr = "actions:"

    i = 0
    numAligned = 0
    totReward = 0.0
    totDiscountedReward = 0.0
    discount = 1.0

    while True:
        qnA = qns.q[0]
        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        #print("node.lang", node.lang, langsVisited.shape)
        langsVisited[0, node.lang] += 1
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        #print("candidates", candidates.Debug())
        _, _, _, _, _, _, _, _, action, link, reward = NeuralWalk(env, params, 0.0, candidates, visited, langsVisited, sess, qnA)
        node = link.childNode
        #print("action", action, qValues)
        actionStr += str(action) + " "

        totReward += reward
        totDiscountedReward += discount * reward

        mainStr += "->" + str(node.lang)
        rewardStr += "->" + str(reward)

        if node.alignedNode is not None:
            mainStr += "*"
            numAligned += 1

        discount *= params.gamma
        i += 1

        if node.urlId == 0:
            break

        if len(visited) > params.maxDocs:
            break

    mainStr += " " + str(i) 
    rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)

    print(actionStr)
    print(mainStr)
    print(rewardStr)
    return ret, totReward, totDiscountedReward

######################################################################################
def Train(params, sess, saver, qns, envs, envsTest):
    print("Start training")
    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        for env in envs:
            TIMER.Start("Trajectory")
            _ = Trajectory(env, epoch, params, sess, qns)
            TIMER.Pause("Trajectory")

        TIMER.Start("Train")
        qns.q[0].corpus.Train(sess, params)
        qns.q[1].corpus.Train(sess, params)
        TIMER.Pause("Train")

        if epoch > 0 and epoch % params.walk == 0:
            print("epoch", epoch)
            SavePlots(sess, qns, params, envs, params.saveDirPlots, epoch, "train")
            SavePlots(sess, qns, params, envsTest, params.saveDirPlots, epoch, "test")

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
    oparser.add_argument("--num-train-hosts", dest="numTrainHosts", type=int,
                         default=1, help="Number of domains to train on")
    oparser.add_argument("--num-test-hosts", dest="numTestHosts", type=int,
                         default=3, help="Number of domains to test on")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    languages = GetLanguages(options.configFile)
    params = LearningParams(languages, options.saveDir, options.saveDirPlots, options.deleteDuplicateTransitions, options.langPair, languages.maxLangId, languages.GetLang("None"))

    if not os.path.exists(options.saveDirPlots): os.mkdir(options.saveDirPlots)

    print("options.numTrainHosts", options.numTrainHosts)
    #hosts = ["http://vade-retro.fr/"]
    hosts = ["http://www.buchmann.ch/", "http://telasmos.org/", "http://tagar.es/"]
    #hosts = ["http://www.visitbritain.com/"]

    #hostsTest = ["http://vade-retro.fr/"]
    #hostsTest = ["http://www.visitbritain.com/"]
    hostsTest = ["http://www.visitbritain.com/", "http://chopescollection.be/", "http://www.bedandbreakfast.eu/"]

    envs = GetEnvs(options.configFile, languages, hosts[:options.numTrainHosts])
    envsTest = GetEnvs(options.configFile, languages, hostsTest[:options.numTestHosts])

    tf.reset_default_graph()
    qns = Qnets(params)
    init = tf.global_variables_initializer()

    saver = None #tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        Train(params, sess, saver, qns, envs, envsTest)

######################################################################################
main()
