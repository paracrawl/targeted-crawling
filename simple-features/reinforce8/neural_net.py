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
    link = candidates.PopWithAction(action)
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

        # mask
        self.numCandidates = tf.placeholder(shape=[None, self.params.NUM_ACTIONS], dtype=tf.float32)
        self.mask = tf.cast(self.numCandidates, dtype=tf.bool)
        self.maskNum = tf.cast(self.mask, dtype=tf.float32)
        self.maskBigNeg = tf.subtract(self.maskNum, 1)
        self.maskBigNeg = tf.multiply(self.maskBigNeg, 999999)

        # graph represention
        self.langsVisited = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        self.numActions = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        
        # link representation
        self.linkSpecificInput = tf.placeholder(shape=[None, self.params.NUM_ACTIONS, self.params.NUM_LINK_FEATURES], dtype=tf.float32)
        print("self.linkSpecific", self.linkSpecificInput.shape)
        self.numLinkFeatures = int(self.linkSpecificInput.shape[2])
        assert(self.numLinkFeatures == params.NUM_LINK_FEATURES)
        #print("self.numLinkFeatures", type(self.numLinkFeatures), self.numLinkFeatures)

        # batch size
        self.batchSize = tf.shape(self.mask)[0]
        
        # network
        self.input = tf.concat([self.langsVisited, self.numActions], 1)
        print("self.input", self.input.shape)

        self.W1 = tf.Variable(tf.random_uniform([3 + 1, params.hiddenDim], minval=0, maxval=0))
        self.b1 = tf.Variable(tf.random_uniform([1, params.hiddenDim], minval=0, maxval=0))
        self.hidden1 = tf.matmul(self.input, self.W1)
        self.hidden1 = tf.add(self.hidden1, self.b1)
        self.hidden1 = tf.nn.relu(self.hidden1)
        #self.hidden1 = tf.nn.sigmoid(self.hidden1)
        print("self.hidden1", self.hidden1.shape)

        self.W2 = tf.Variable(tf.random_uniform([params.hiddenDim, params.linkDim], minval=0, maxval=0))
        self.b2 = tf.Variable(tf.random_uniform([1, params.linkDim], minval=0, maxval=0))
        self.hidden2 = tf.matmul(self.hidden1, self.W2)
        self.hidden2 = tf.add(self.hidden2, self.b2)
        #self.hidden2 = tf.nn.relu(self.hidden3)
        #self.hidden2 = tf.nn.sigmoid(self.hidden3)
        print("self.hidden2", self.hidden2.shape)

        # link-specific
        self.WlinkSpecific = tf.Variable(tf.random_uniform([self.numLinkFeatures, params.linkDim], 0, 0.01))
        self.blinkSpecific = tf.Variable(tf.random_uniform([1, params.linkDim], 0, 0.01))

        #print("self.linkSpecific1", self.linkSpecific.shape)
        self.linkSpecific = tf.reshape(self.linkSpecificInput, [self.batchSize * self.params.NUM_ACTIONS, self.numLinkFeatures ])
        #print("self.linkSpecific2", self.linkSpecific.shape)

        self.linkSpecific = tf.matmul(self.linkSpecific, self.WlinkSpecific)
        self.linkSpecific = tf.add(self.linkSpecific, self.blinkSpecific)        
        self.linkSpecific = tf.nn.relu(self.linkSpecific)
        #self.linkSpecific = tf.nn.sigmoid(self.linkSpecific)
        #print("self.linkSpecific3", self.linkSpecific.shape)
        
        self.WlinkSpecific2 = tf.Variable(tf.random_uniform([params.linkDim, params.linkDim], 0, 0.01))
        self.blinkSpecific2 = tf.Variable(tf.random_uniform([1, params.linkDim], 0, 0.01))

        self.linkSpecific = tf.matmul(self.linkSpecific, self.WlinkSpecific2)
        self.linkSpecific = tf.add(self.linkSpecific, self.blinkSpecific2)

        #print("self.linkSpecific4", self.linkSpecific.shape)
        self.linkSpecific = tf.reshape(self.linkSpecific, [self.batchSize, self.params.NUM_ACTIONS, params.linkDim])
        print("self.linkSpecific5", self.linkSpecific.shape)

        # final q-values
        self.logit = tf.reshape(self.hidden2, [self.batchSize, 1, params.linkDim])
        self.logit = tf.multiply(self.linkSpecific, self.logit)
        self.logit = tf.reduce_sum(self.logit, axis=2)

        # softmax
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
        self.r2 = self.params.NUM_ACTIONS
        self.r3 = self.r1 * self.r2          # 0 2 4 6 8
        self.indexes = self.r3 + self.action_holder # r3 + 0/1 offset depending on action
       
        self.o1 = tf.reshape(self.probs, [-1]) # all action probs in 1-d
        self.responsible_outputs = tf.gather(self.o1, self.indexes) # the prob of the action. Should have just stored it!? len=length of trajectory

        self.l1 = tf.log(self.responsible_outputs)
        self.l2 = self.l1 * self.reward_holder  # log prob * reward. len=length of trajectory
        self.loss = -tf.reduce_mean(self.l2)    # 1 number
        
        # calc grads, but don't actually update weights
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars): # idx = contiguous int, var = variable shape (4,8) (8,2)
            #print("idx", idx)
            #print("var", var)
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        self.gradients = tf.gradients(self.loss,tvars) # grads same shape as gradient_holder0. (4,8) (8,2)

        #self.trainer = tf.train.GradientDescentOptimizer(learning_rate=params.lrn_rate)
        #self.trainer = tf.train.AdagradOptimizer(learning_rate=params.lrn_rate)        
        self.trainer = tf.train.AdamOptimizer(learning_rate=params.lrn_rate)        
        #self.updateModel = self.trainer.minimize(self.loss)
        self.update_batch = self.trainer.apply_gradients(zip(self.gradient_holders,tvars))

    def PredictAll(self, env, sess, langIds, visited, candidates):
        numActions, numCandidates, linkSpecific = candidates.GetMask()
        #print("numActions", numActions)
        #print("numCandidates", numCandidates.shape, numCandidates)
        #print("linkSpecific", linkSpecific.shape, linkSpecific)
        assert(numActions > 0)

        numActionsArr = np.empty([1,1])
        numActionsArr[0,0] = numActions

        langsVisited = GetLangsVisited(visited, langIds, env)
        #print("langsVisited", langsVisited)
        
        (probs,logit, smNumer, smNumerSum, maxLogit, maskNum, maskBigNeg) = sess.run([self.probs, self.logit, self.smNumer, self.smNumerSum, self.maxLogit, self.maskNum, self.maskBigNeg], 
                                feed_dict={self.numCandidates: numCandidates,
                                        self.linkSpecificInput: linkSpecific,
                                        self.langsVisited: langsVisited,
                                        self.numActions: numActionsArr})
        probs = np.reshape(probs, [probs.shape[1] ])        
        try:
            action = np.random.choice(self.params.NUM_ACTIONS,p=probs)
        except:
            print("langsVisited", probs, logit, smNumer, smNumerSum, langsVisited)
            print("probs", probs)
            print("logit", logit)
            print("maxLogit", maxLogit)
            print("smNumer", smNumer)
            print("smNumerSum", smNumerSum)
            print("langsVisited", langsVisited)
            print("numCandidates", numCandidates)
            print("maskBigNeg", maskBigNeg)
            bugger_something_went_wrong

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

        #print("action", action, probs, logit, mask, langsVisited, numActions)
        if np.random.rand(1) < .005:
            print("action", action, probs, logit, numCandidates, linkSpecific.tolist(), langsVisited, numActions)
        #print()

        return action

    def GetGradBuffer(self, sess):
        gradBuffer = sess.run(tf.trainable_variables())
        for idx,grad in enumerate(gradBuffer):
            #print("idx", idx)
            gradBuffer[idx] = grad * 0
        #print("gradBuffer", gradBuffer)
        return gradBuffer

    def CalcDiscountedReward(self, transitions):
        runningReward = 0.0
        for t in reversed(range(0, len(transitions))):
            transition = transitions[t]
            runningReward = runningReward * self.params.gamma + transition.reward
            transition.discountedReward = runningReward
            #print("t", t, transition.Debug())

    def CalcGrads(self, sess, corpus):
        #print("CalcGrads")
        self.CalcDiscountedReward(corpus.transitions)

        #for transition in self.transitions:
        #    print(transition.Debug())
        #lastTrans = self.transitions[-1]
        #print("lastTrans", lastTrans.Debug())

        batchSize = len(corpus.transitions)
        #print("batchSize", batchSize)
        numActions = np.empty([batchSize, 1], dtype=np.int)
        numCandidates = np.empty([batchSize, self.params.NUM_ACTIONS], dtype=np.float)
        #print("numActions", numActions, numCandidates)

        langIds = np.empty([batchSize, 2], dtype=np.int)
        langsVisited = np.empty([batchSize, 3])

        actions = np.empty([batchSize], dtype=np.int)
        discountedRewards = np.empty([batchSize], dtype=np.float32)

        linkSpecific = np.empty([batchSize, self.params.NUM_ACTIONS, self.params.NUM_LINK_FEATURES], dtype=np.float32)

        i = 0
        for transition in corpus.transitions:
            #curr = transition.curr
            #next = transition.next
            #print("transition.numActions", transition.numActions, transition.targetQ.shape, transition.candidates.Count())
            numActions[i, 0] = transition.numActions
            numCandidates[i, :] = transition.numCandidates

            langIds[i, :] = transition.langIds
            langsVisited[i, :] = transition.langsVisited

            actions[i] = transition.action
            discountedRewards[i] = transition.discountedReward

            linkSpecific[i, :, :] = transition.linkSpecific

            i += 1

        (loss, W1, b1, grads) = sess.run([self.loss, self.W1, self.b1, self.gradients], 
                                    feed_dict={self.numCandidates: numCandidates,
                                            self.linkSpecificInput: linkSpecific,
                                            self.langsVisited: langsVisited,
                                            self.numActions: numActions,
                                            self.action_holder: actions,
                                            self.reward_holder: discountedRewards})
        #print("loss", loss, numActions)
        #print("W1", W1, b1)
        #print("grads", grads)
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

        for idx,grad in enumerate(grads):
            #print("idx", idx)
            corpus.gradBuffer[idx] += grad         # accumulate gradients

        corpus.transitions.clear()

        #print("loss", loss)
        return loss

    def UpdateGrads(self, sess, corpus):
        #print("UpdateGrads")

        feed_dict= dict(zip(self.gradient_holders, corpus.gradBuffer))
        _ = sess.run(self.update_batch, feed_dict=feed_dict)

        for idx,grad in enumerate(corpus.gradBuffer):
            corpus.gradBuffer[idx] = grad * 0
