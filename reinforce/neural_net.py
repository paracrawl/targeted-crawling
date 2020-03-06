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
        _, parentLang, _, numSiblings, numVisitedSiblings, numMatchedSiblings, parentMatched, linkLang = candidates.GetFeatures()
        parentLang1 = parentLang[0, action]
        numSiblings1 = numSiblings[0, action]
        numVisitedSiblings1 = numVisitedSiblings[0, action]
        numMatchedSiblings1 = numMatchedSiblings[0, action]
        parentMatched1 = parentMatched[0, action]
        linkLang1 = linkLang[0, action]
        key = (parentLang1, numSiblings1, numVisitedSiblings1, numMatchedSiblings1, parentMatched1, linkLang1)
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
def NeuralWalk(env, params, eps, candidates, visited, sess, qnA):
    action = qnA.PredictAll(env, sess, params.langIds, visited, candidates)
    link, reward = GetNextState(env, params, action, visited, candidates)
    assert(link is not None)

    return action, link, reward

######################################################################################
class Qnetwork():
    def __init__(self, params):
        self.params = params
        self.corpus = Corpus(params, self)

        HIDDEN_DIM = 512

        # mask
        self.mask = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.bool)
        self.maskNum = tf.cast(self.mask, dtype=tf.float32)
        
        # graph represention
        self.langIds = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.langsVisited = tf.placeholder(shape=[None, 6], dtype=tf.float32)
        self.numActions = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # link representation
        self.parentLang = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numVisitedSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.numMatchedSiblings = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.parentMatched = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)
        self.linkLang = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.float32)

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
        #self.hidden3 = tf.add(self.hidden3, self.b3)
        self.hidden3 = tf.nn.relu(self.hidden3)
        self.hidden3 = tf.nn.sigmoid(self.hidden3)

        # link-specific
        self.WlinkSpecific = tf.Variable(tf.random_uniform([6, HIDDEN_DIM], 0, 0.01))
        self.blinkSpecific = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], 0, 0.01))

        self.linkSpecific = tf.stack([tf.transpose(self.parentLang), 
                                    tf.transpose(self.numSiblings), 
                                    tf.transpose(self.numVisitedSiblings),
                                    tf.transpose(self.numMatchedSiblings),
                                    tf.transpose(self.parentMatched),
                                    tf.transpose(self.linkLang)], 0)
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

        # softmax
        self.logit = self.hidden3
        self.maxLogit = tf.multiply(self.logit, self.maskNum)
        self.maxLogit = tf.reduce_max(self.maxLogit, axis=1)
        self.maxLogit = tf.reshape(self.maxLogit, [self.batchSize, 1])
        self.smNumer = tf.subtract(self.hidden3, self.maxLogit)
        self.smNumer = tf.exp(self.smNumer)
        self.smNumer = tf.multiply(self.smNumer, self.maskNum)
        self.smNumerSum = tf.reduce_sum(self.smNumer, axis=1)
        self.smNumerSum = tf.reshape(self.smNumerSum, [self.batchSize, 1])
        
        self.probs = tf.divide(self.smNumer, self.smNumerSum)
        self.chosenAction = tf.argmax(self.probs,0)

        # training
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32) #  0 or 1

        self.r1 = tf.range(0, self.batchSize)  # 0 1 2 3 4 len=length of trajectory
        self.r2 = self.params.MAX_NODES
        self.r3 = self.r1 * self.r2          # 0 2 4 6 8
        self.indexes = self.r3 + self.action_holder # r3 + 0/1 offset depending on action
       
        self.o1 = tf.reshape(self.probs, [-1]) # all action probs in 1-d
        self.responsible_outputs = tf.gather(self.o1, self.indexes) # the prob of the action. Should have just stored it!? len=length of trajectory

        self.l1 = tf.log(self.responsible_outputs)
        self.l2 = self.l1 * self.reward_holder  # log prob * reward. len=length of trajectory
        self.loss = -tf.reduce_mean(self.l2)    # 1 number
        
        #self.trainer = tf.train.GradientDescentOptimizer(learning_rate=params.lrn_rate)
        #self.trainer = tf.train.AdagradOptimizer(learning_rate=params.lrn_rate)        
        self.trainer = tf.train.AdamOptimizer(learning_rate=params.lrn_rate)        
        self.updateModel = self.trainer.minimize(self.loss)

    def PredictAll(self, env, sess, langIds, visited, candidates):
        numActions, parentLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, parentMatched, linkLang = candidates.GetFeatures()
        #print("numActions", numActions)
        
        numActionsNP = np.empty([1,1], dtype=np.int32)
        numActionsNP[0,0] = numActions

        assert(numActions > 0)

        #print("parentLang", numActions, parentLang.shape)
        #print("mask", mask.shape, mask)
        langsVisited = GetLangsVisited(visited, langIds, env)
        #print("langsVisited", langsVisited)
        
        (probs,) = sess.run([self.probs], 
                                feed_dict={self.parentLang: parentLang,
                                    self.numActions: numActionsNP,
                                    self.mask: mask,
                                    self.numSiblings: numSiblings,
                                    self.numVisitedSiblings: numVisitedSiblings,
                                    self.numMatchedSiblings: numMatchedSiblings,
                                    self.parentMatched: parentMatched,
                                    self.linkLang: linkLang,
                                    self.langIds: langIds,
                                    self.langsVisited: langsVisited})
        #print("hidden3", hidden3.shape, hidden3)
        #print("qValues", qValues.shape, qValues)
        #print("   maxQ", maxQ.shape, maxQ)
        #print("  probs", probs.shape, probs)
        #print("  chosenAction", chosenAction.shape, chosenAction)
        #print("linkSpecific", linkSpecific.shape)
        #print("numSiblings", numSiblings.shape)
        #print("numVisitedSiblings", numVisitedSiblings.shape)
        #print("numMatchedSiblings", numMatchedSiblings.shape)
        #print("   qValues", qValues)
        #print()

        probs = np.reshape(probs, [probs.shape[1] ])
        action = np.random.choice(probs,p=probs)
        #print("  action", action)
        action = np.argmax(probs == action)
        #print("  action", action)

        return action

    def Update(self, sess, numActions, parentLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, parentMatched, linkLang, langIds, langsVisited, actions, discountedRewards):
        #print("actions, discountedRewards", actions, discountedRewards)
        #print("input", parentLang.shape, langIds.shape, langFeatures.shape, targetQ.shape)
        #print("targetQ", targetQ)
        (_, loss) = sess.run([self.updateModel, self.loss], 
                                    feed_dict={self.parentLang: parentLang, 
                                            self.numActions: numActions,
                                            self.mask: mask,
                                            self.numSiblings: numSiblings,
                                            self.numVisitedSiblings: numVisitedSiblings,
                                            self.numMatchedSiblings: numMatchedSiblings,
                                            self.parentMatched: parentMatched,
                                            self.linkLang: linkLang,
                                            self.langIds: langIds, 
                                            self.langsVisited: langsVisited,
                                            self.action_holder: actions,
                                            self.reward_holder: discountedRewards})
        #print("loss", loss, numActions)
        #print("hidden3", hidden3.shape, hidden3)
        #print("   qValues", qValues.shape, qValues)
        #print("   maskNum", maskNum.shape, maskNum)
        #print("   maskNumNeg", maskNumNeg.shape, maskNumNeg)
        #print("   maxQ", maxQ.shape, maxQ)
        #print("   smNumer", smNumer.shape, smNumer)
        #print("   smNumerSum", smNumerSum.shape, smNumerSum)
        #print("   probs", probs.shape, probs)
        #print("   o1", o1.shape, o1)
        #print("   indexes", indexes.shape, indexes)
        #print("   responsible_outputs", responsible_outputs.shape, responsible_outputs)
        #print()

        return loss

