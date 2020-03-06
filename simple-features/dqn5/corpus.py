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
        parentLang = np.empty([batchSize, self.params.MAX_NODES], dtype=np.int)
        mask = np.empty([batchSize, self.params.MAX_NODES], dtype=np.bool)
        
        langIds = np.empty([batchSize, 2], dtype=np.int)
        targetQ = np.empty([batchSize, self.params.MAX_NODES])

        langsVisited = np.empty([batchSize, 6])
        
        i = 0
        for transition in batch:
            #curr = transition.curr
            #next = transition.next
            #print("transition.numActions", transition.numActions, transition.targetQ.shape, transition.candidates.Debug())
            assert(transition.numActions == transition.targetQ.shape[1])
            numActions[i, 0] = transition.numActions
            parentLang[i, :] = transition.parentLang
            mask[i, :] = transition.mask

            langIds[i, :] = transition.langIds
            targetQ[i, 0:transition.numActions] = transition.targetQ

            langsVisited[i, :] = transition.langsVisited

            i += 1

        loss = self.qn.Update(sess, numActions, parentLang, mask, langIds, langsVisited, targetQ)

        #print("loss", loss)
        return loss

