import os
import sys
import numpy as np
import tensorflow as tf

from corpus import Corpus

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from helpers import Link

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
        self.langsVisited = tf.placeholder(shape=[None, 3], dtype=tf.float32)
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

        self.W1 = tf.Variable(tf.random_uniform([3 + 3, HIDDEN_DIM], 0, 0.01))
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

