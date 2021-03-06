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
        langsVisited = np.empty([batchSize, 3])
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

        loss = self.qn.Update(sess, numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, langIds, langsVisited, targetQ)

        #print("loss", loss)
        return loss

