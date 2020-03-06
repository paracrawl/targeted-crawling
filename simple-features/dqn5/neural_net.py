import os
import sys
import numpy as np
import tensorflow as tf

from corpus import Corpus

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from helpers import Link
from candidate import GetLangsVisited

######################################################################################
def GetNextState(env, params, action, visited, candidates):

    #print("candidates", action, candidates.Debug())
    if action == -1:
        # no explicit stop state but no candidates
        stopNode = env.nodes[0]
        link = Link("", 0, stopNode, stopNode)
    else:
        _, parentLang, _ = candidates.GetFeatures()
        parentLang1 = parentLang[0, action]
        key = (parentLang1, )
        link = candidates.Pop(key)
        candidates.AddLinks(link.childNode, visited, params)

    assert(link.childNode.urlId not in visited)
    visited.add(link.childNode.urlId)
 
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
def NeuralWalk(env, params, candidates, visited, sess, qnA):
    qValues, maxQ, action = qnA.PredictAll(env, sess, params.langIds, visited, candidates)

    #print("action", action, parentLang, qValues)
    if action >= 0:
        if np.random.rand(1) < params.eps:
            #print("actions", type(actions), actions)
            numActions, _, _ = candidates.GetFeatures()
            action = np.random.randint(0, numActions)
            maxQ = qValues[0, action]
            #print("random")
        #print("action", action, qValues)

    #print("action", action, maxQ, qValues)
    link, reward = GetNextState(env, params, action, visited, candidates)
    assert(link is not None)
    #print("action", action, qValues, link.childNode.Debug(), reward)

    return qValues, maxQ, action, link, reward

######################################################################################
class Qnets():
    def __init__(self, params):
        self.q = []
        self.q.append(Qnetwork(params))
        self.q.append(Qnetwork(params))

######################################################################################
class Qnetwork():
    def __init__(self, params):
        self.params = params
        self.corpus = Corpus(params, self)

        HIDDEN_DIM = 512

        # mask
        self.mask = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.bool)

        # graph represention
        self.langIds = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.langsVisited = tf.placeholder(shape=[None, 6], dtype=tf.float32)
        self.numActions = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # link representation
        self.parentLang = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)

        # batch size
        self.batchSize = tf.shape(self.parentLang)[0]
        
        # network
        self.input = tf.concat([self.langIds, self.langsVisited, self.numActions], 1)
        #print("self.input", self.input.shape)

        self.W1 = tf.Variable(tf.random_uniform([2 + 6 + 1, HIDDEN_DIM], 0, 0.01))
        self.b1 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden1 = tf.matmul(self.input, self.W1)
        self.hidden1 = tf.add(self.hidden1, self.b1)
        self.hidden1 = tf.nn.relu(self.hidden1)
        #self.hidden1 = tf.nn.sigmoid(self.hidden1)

        self.W2 = tf.Variable(tf.random_uniform([HIDDEN_DIM, HIDDEN_DIM], 0, 0.01))
        self.b2 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden2 = tf.matmul(self.hidden1, self.W2)
        self.hidden2 = tf.add(self.hidden2, self.b2)
        self.hidden2 = tf.nn.relu(self.hidden2)
        #self.hidden2 = tf.nn.sigmoid(self.hidden2)
        
        self.W3 = tf.Variable(tf.random_uniform([HIDDEN_DIM, HIDDEN_DIM], 0, 0.01))
        self.b3 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.hidden3 = tf.matmul(self.hidden2, self.W3)
        self.hidden3 = tf.add(self.hidden3, self.b3)
        self.hidden3 = tf.nn.relu(self.hidden3)
        #self.hidden3 = tf.nn.sigmoid(self.hidden3)

        # link-specific
        self.WlinkSpecific = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))
        self.blinkSpecific = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))

        self.linkSpecific = tf.stack([tf.transpose(self.parentLang)], 0)
        self.linkSpecific = tf.transpose(self.linkSpecific)
        self.linkSpecific = tf.reshape(self.linkSpecific, [self.batchSize * self.params.MAX_NODES, self.linkSpecific.shape[2] ])
 
        self.linkSpecific = tf.matmul(self.linkSpecific, self.WlinkSpecific)
        self.linkSpecific = tf.add(self.linkSpecific, self.blinkSpecific)        
        self.linkSpecific = tf.nn.relu(self.linkSpecific)
        #self.linkSpecific = tf.nn.sigmoid(self.linkSpecific)

        self.linkSpecific = tf.reshape(self.linkSpecific, [self.batchSize, self.params.MAX_NODES, 512])
        
        # final q-values
        self.hidden3 = tf.reshape(self.hidden3, [self.batchSize, 1, HIDDEN_DIM])
        self.hidden3 = tf.multiply(self.linkSpecific, self.hidden3)
        self.hidden3 = tf.reduce_sum(self.hidden3, axis=2)

        self.qValues = tf.boolean_mask(self.hidden3, self.mask, axis=0)

        #self.hidden3 = tf.math.reduce_sum(self.hidden3, axis=1)
        #self.qValues = self.hidden3
        #print("self.qValues", self.qValue.shapes)
       
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        nextQMasked = tf.boolean_mask(self.nextQ, self.mask, axis=0)

        self.loss = nextQMasked - self.qValues
        #self.loss = tf.reduce_max(tf.square(self.loss))
        self.loss = tf.reduce_mean(tf.square(self.loss))
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

    def PredictAll(self, env, sess, langIds, visited, candidates):
        numActions, parentLang, mask = candidates.GetFeatures()
        assert(numActions > 0)
        
        numActionsNP = np.empty([1,1], dtype=np.int32)
        numActionsNP[0,0] = numActions

        #print("parentLang", numActions, parentLang.shape)
        #print("mask", mask.shape, mask)
        #print("linkLang", linkLang.shape, linkLang)
        langsVisited = GetLangsVisited(visited, langIds, env)
        #print("langsVisited", langsVisited)
        
        (qValues, ) = sess.run([self.qValues, ], 
                                feed_dict={self.parentLang: parentLang,
                                    self.numActions: numActionsNP,
                                    self.mask: mask,
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

        return qValues, maxQ, action

    def Update(self, sess, numActions, parentLang, mask, langIds, langsVisited, targetQ):
        #print("input", parentLang.shape, langIds.shape, langFeatures.shape, targetQ.shape)
        #print("targetQ", targetQ)
        _, loss = sess.run([self.updateModel, self.loss], 
                                    feed_dict={self.parentLang: parentLang, 
                                            self.numActions: numActions,
                                            self.mask: mask,
                                            self.langIds: langIds, 
                                            self.langsVisited: langsVisited,
                                            self.nextQ: targetQ})
        #print("loss", loss, numActions)
        return loss

