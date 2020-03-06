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
        #_, parentLang, _ = candidates.GetFeatures()
        #parentLang1 = parentLang[0, action]
        #key = (parentLang1,)
        key = (action,)
        
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

        HIDDEN_DIM = 4

        # mask
        self.mask = tf.placeholder(shape=[None, self.params.MAX_NODES], dtype=tf.bool)
        self.maskNum = tf.cast(self.mask, dtype=tf.float32)
        self.maskBigNeg = tf.subtract(self.maskNum, 1)
        self.maskBigNeg = tf.multiply(self.maskBigNeg, 999999)

        # graph represention
        self.langsVisited = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        
        # link representation

        # batch size
        self.batchSize = tf.shape(self.mask)[0]
        
        # network
        self.input = tf.concat([self.langsVisited], 1)
        #print("self.input", self.input.shape)

        self.W1 = tf.Variable(tf.random_uniform([3, HIDDEN_DIM], minval=0, maxval=0))
        self.b1 = tf.Variable(tf.random_uniform([1, HIDDEN_DIM], minval=0, maxval=0))
        self.hidden1 = tf.matmul(self.input, self.W1)
        self.hidden1 = tf.add(self.hidden1, self.b1)
        self.hidden1 = tf.nn.relu(self.hidden1)
        #self.hidden1 = tf.nn.sigmoid(self.hidden1)
        print("self.hidden1", self.hidden1.shape)

        self.W2 = tf.Variable(tf.random_uniform([HIDDEN_DIM, params.MAX_NODES], minval=0, maxval=0))
        self.b2 = tf.Variable(tf.random_uniform([1, params.MAX_NODES], minval=0, maxval=0))
        self.hidden2 = tf.matmul(self.hidden1, self.W2)
        self.hidden2 = tf.add(self.hidden2, self.b2)
        #self.hidden2 = tf.nn.relu(self.hidden3)
        #self.hidden2 = tf.nn.sigmoid(self.hidden3)
        print("self.hidden2", self.hidden2.shape)

        # link-specific

        # final q-values
        #self.hidden2 = tf.reshape(self.hidden2, [self.batchSize, 1, HIDDEN_DIM])
        #self.hidden2 = tf.reduce_sum(self.hidden2, axis=2)

        # softmax
        self.logit = self.hidden2
        self.maxLogit = tf.add(self.logit, self.maskBigNeg)
        self.maxLogit = tf.reduce_max(self.maxLogit, axis=1)
        self.maxLogit = tf.reshape(self.maxLogit, [self.batchSize, 1])

        self.smNumer = tf.subtract(self.logit, self.maxLogit)
        self.smNumer = tf.multiply(self.smNumer, self.maskNum)
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
        numActions, mask = candidates.GetFeatures()
        #print("numActions", numActions)
        #print("mask", mask.shape, mask)
        #print("parentLang", parentLang.shape, parentLang)
        assert(numActions > 0)

        langsVisited = GetLangsVisited(visited, langIds, env)
        #print("langsVisited", langsVisited)
        
        (probs,logit, smNumer, smNumerSum, maxLogit, maskBigNeg) = sess.run([self.probs, self.logit, self.smNumer, self.smNumerSum, self.maxLogit, self.maskBigNeg], 
                                feed_dict={self.mask: mask,
                                    self.langsVisited: langsVisited})
        probs = np.reshape(probs, [probs.shape[1] ])        
        try:
            action = np.random.choice(self.params.MAX_NODES,p=probs)
        except:
            print("langsVisited", probs, logit, smNumer, smNumerSum, langsVisited)
            print("probs", probs)
            print("logit", logit)
            print("maxLogit", maxLogit)
            print("smNumer", smNumer)
            print("smNumerSum", smNumerSum)
            print("langsVisited", langsVisited)
            print("mask", mask)
            print("maskBigNeg", maskBigNeg)
            dsds

        # print("langsVisited", probs, logit, smNumer, smNumerSum, langsVisited)
        # print("probs", probs)
        # print("logit", logit)
        # print("maxLogit", maxLogit)
        # print("smNumer", smNumer)
        # print("smNumerSum", smNumerSum)
        # print("langsVisited", langsVisited)
        # print("mask", mask)
        # print("maskBigNeg", maskBigNeg)
        # print()

        #print("action", action, probs, logit, mask, langsVisited, parentLang, numActions)
        if np.random.rand(1) < .005:
            print("action", action, probs, logit, mask, langsVisited, numActions)
        #print()

        return action

    def Update(self, sess, numActions, mask, langIds, langsVisited, actions, discountedRewards):
        #print("actions, discountedRewards", actions, discountedRewards)
        #print("input", parentLang.shape, langIds.shape, langFeatures.shape, targetQ.shape)
        #print("targetQ", targetQ)
        (_, loss, W1, b1) = sess.run([self.updateModel, self.loss, self.W1, self.b1], 
                                    feed_dict={self.mask: mask,
                                            self.langsVisited: langsVisited,
                                            self.action_holder: actions,
                                            self.reward_holder: discountedRewards})
        #print("loss", loss, numActions)
        #print("W1", W1, b1)
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

