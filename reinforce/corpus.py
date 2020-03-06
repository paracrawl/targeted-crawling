import os
import sys
import numpy as np

######################################################################################
class Corpus:
    def __init__(self, params, qn):
        self.params = params
        self.qn = qn
        self.transitions = []
        self.losses = []

    def AddTransition(self, transition):    
        self.transitions.append(transition)

    def GetBatchWithoutDelete(self, maxBatchSize):
        batch = []

        size = len(self.transitions)
        for i in range(maxBatchSize):
            idx = np.random.randint(0, size)
            transition = self.transitions[idx]
            batch.append(transition)

        return batch

    def CalcDiscountedReward(self):
        runningReward = 0.0
        for t in reversed(range(0, len(self.transitions))):
            transition = self.transitions[t]
            runningReward = runningReward * self.params.gamma + transition.reward
            transition.discountedReward = runningReward
            #print("t", t, transition.Debug())

    def Train(self, sess, params):
        if len(self.transitions) >= params.minCorpusSize:
            self.CalcDiscountedReward()
            #for transition in self.transitions:
            #    print(DebugTransition(transition))

            batch = []
            for idx in range(0, len(self.transitions)):
                transition = self.transitions[idx]
                batch.append(transition)
                if len(batch) >= params.maxBatchSize:
                    loss = self.UpdateQN(params, sess, batch)
                    self.losses.append(loss)
                    batch = []

            self.transitions.clear()
        
    def UpdateQN(self, params, sess, batch):
        batchSize = len(batch)
        #print("batchSize", batchSize)
        numActions = np.empty([batchSize, 1], dtype=np.int)
        parentLang = np.empty([batchSize, self.params.MAX_NODES], dtype=np.int)
        mask = np.empty([batchSize, self.params.MAX_NODES], dtype=np.bool)
        numSiblings = np.empty([batchSize, self.params.MAX_NODES], dtype=np.float32)
        numVisitedSiblings = np.empty([batchSize, self.params.MAX_NODES], dtype=np.float32)
        numMatchedSiblings = np.empty([batchSize, self.params.MAX_NODES], dtype=np.float32)
        parentMatched = np.empty([batchSize, self.params.MAX_NODES], dtype=np.float32)
        linkLang = np.empty([batchSize, self.params.MAX_NODES], dtype=np.float32)

        langIds = np.empty([batchSize, 2], dtype=np.int)
        langsVisited = np.empty([batchSize, 6])

        actions = np.empty([batchSize], dtype=np.int)
        discountedRewards = np.empty([batchSize], dtype=np.float32)
        
        i = 0
        for transition in batch:
            #curr = transition.curr
            #next = transition.next
            #print("transition.numActions", transition.numActions, transition.targetQ.shape, transition.candidates.Count())
            numActions[i, 0] = transition.numActions
            parentLang[i, :] = transition.parentLang
            mask[i, :] = transition.mask
            numSiblings[i, :] = transition.numSiblings
            numVisitedSiblings[i, :] = transition.numVisitedSiblings
            numMatchedSiblings[i, :] = transition.numMatchedSiblings
            parentMatched[i, :] = transition.parentMatched
            linkLang[i, :] = transition.linkLang

            langIds[i, :] = transition.langIds
            langsVisited[i, :] = transition.langsVisited

            actions[i] = transition.action
            discountedRewards[i] = transition.discountedReward

            i += 1

        loss = self.qn.Update(sess, numActions, parentLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, parentMatched, linkLang, langIds, langsVisited, actions, discountedRewards)

        #print("loss", loss)
        return loss

