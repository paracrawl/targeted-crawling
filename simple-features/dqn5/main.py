#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import tensorflow as tf

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from common import GetLanguages, Languages, Timer
from helpers import GetEnvs, GetVistedSiblings, GetMatchedSiblings, NumParallelDocs, Env, Link
from corpus import Corpus
from neural_net import Qnets, Qnetwork, NeuralWalk, GetNextState
from candidate import Candidates, GetLangsVisited
from save_plot import SavePlot

######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, saveDirPlots, deleteDuplicateTransitions, langPair, maxLangId, defaultLang):
        self.gamma = 1.0 #0.99
        self.lrn_rate = 0.001
        self.alpha = 0.7
        self.max_epochs = 100001
        self.eps = 0.1 #1.0
        self.maxBatchSize = 32
        self.minCorpusSize = 200
        self.overSampling = 10
        
        self.debug = False
        self.walk = 10

        self.saveDir = saveDir
        self.saveDirPlots = saveDirPlots

        self.deleteDuplicateTransitions = deleteDuplicateTransitions
        
        self.reward = 1.0 #17.0
        self.cost = 0.0 #-1.0
        self.unusedActionCost = 0.0 #-555.0
        self.maxDocs = 500 #9999999999

        self.maxLangId = maxLangId
        self.defaultLang = defaultLang
        self.MAX_NODES = 3

        langPairList = langPair.split(",")
        assert(len(langPairList) == 2)
        self.langIds = np.empty([1,2], dtype=np.int32)
        self.langIds[0,0] = languages.GetLang(langPairList[0])
        self.langIds[0,1] = languages.GetLang(langPairList[1])
        #print("self.langs", self.langs)

######################################################################################
def RunRLSavePlots(sess, qns, params, envs, saveDirPlots, epoch, sset):
    for env in envs:
        RunRLSavePlot(sess, qns, params, env, saveDirPlots, epoch, sset)

def RunRLSavePlot(sess, qn, params, env, saveDirPlots, epoch, sset):
    arrRL, totReward, totDiscountedReward = Trajectory(env, params, sess, qn, True, 1)
    SavePlot(params, env, saveDirPlots, epoch, sset, arrRL, totReward, totDiscountedReward)

######################################################################################
class Transition:
    def __init__(self, env, action, link, langIds, targetQ, visited, candidates, nextVisited, nextCandidates):
        self.action = action
        self.link = link

        self.langIds = langIds 
        self.targetQ = np.array(targetQ, copy=True)

        if visited is not None:
            self.visited = visited
            self.langsVisited = GetLangsVisited(visited, langIds, env)

        if candidates is not None:
            self.candidates = candidates
            numActions, parentLang, mask = candidates.GetFeatures()
            self.numActions = numActions
            self.parentLang = np.array(parentLang, copy=True) 
            self.mask = np.array(mask, copy=True) 
            
        self.nextVisited = nextVisited
        self.nextCandidates = nextCandidates

    def Debug(self):
        ret = str(self.link.parentNode.urlId) + "->" + str(self.link.childNode.urlId) + " " + str(self.visited)
        return ret
    
######################################################################################
def Neural(env, params, prevTransition, sess, qnA, qnB):
    nextCandidates = prevTransition.nextCandidates.copy()
    nextVisited = prevTransition.nextVisited.copy()

    qValues, maxQ, action, link, reward = NeuralWalk(env, params, nextCandidates, nextVisited, sess, qnA)
    assert(link is not None)
    assert(qValues.shape[1] > 0)
    #print("qValues", qValues.shape, action, prevTransition.nextCandidates.Count(), nextCandidates.Count())
    nextCandidates.Group(nextVisited)

    # calc nextMaxQ
    if nextCandidates.Count() > 0:
        #  links to follow NEXT
        _, _, nextAction = qnA.PredictAll(env, sess, params.langIds, nextVisited, nextCandidates)
        #print("nextAction", nextAction, nextLangRequested, nextCandidates.Debug())
        nextQValuesB, _, _ = qnB.PredictAll(env, sess, params.langIds, nextVisited, nextCandidates)
        nextMaxQ = nextQValuesB[0, nextAction]
        #print("nextMaxQ", nextMaxQ, nextMaxQB, nextQValuesA[0, nextAction])
    else:
        nextMaxQ = 0

    newVal = reward + params.gamma * nextMaxQ
    targetQ = (1 - params.alpha) * maxQ + params.alpha * newVal
    qValues[0, action] = targetQ

    transition = Transition(env,
                            action, 
                            link,
                            params.langIds,
                            qValues,
                            prevTransition.nextVisited,
                            prevTransition.nextCandidates,
                            nextVisited,
                            nextCandidates)

    return transition, reward

######################################################################################
def Trajectory(env, params, sess, qns, test, verbose):
    ret = []
    totReward = 0.0
    totDiscountedReward = 0.0
    discount = 1.0

    startNode = env.nodes[sys.maxsize]

    nextVisited = set()
    nextVisited.add(startNode.urlId)

    nextCandidates = Candidates(params, env)
    nextCandidates.AddLinks(startNode, nextVisited, params)
    nextCandidates.Group(nextVisited)

    transition = Transition(env, -1, None, params.langIds, 0, None, None, nextVisited, nextCandidates)
    #print("candidates", transition.nextCandidates.Debug())

    if verbose > 0:
        mainStr = "lang:" + str(startNode.lang)
        rewardStr = "rewards:"
        actionStr = "actions:"

    while True:
        tmp = np.random.rand(1)
        if tmp > 0.5:
            qnA = qns.q[0]
            qnB = qns.q[1]
        else:
            qnA = qns.q[1]
            qnB = qns.q[0]

        transition, reward = Neural(env, params, transition, sess, qnA, qnB)
        #print("visited", transition.visited)
        #print("candidates", transition.nextCandidates.Debug())
        #print("transition", transition.Debug())
        #print()

        numParallelDocs = NumParallelDocs(env, transition.nextVisited)
        ret.append(numParallelDocs)

        totReward += reward
        totDiscountedReward += discount * reward
        discount *= params.gamma

        if verbose > 0:
            mainStr += "->" + str(transition.link.childNode.lang)
            rewardStr += "->" + str(reward)
            actionStr += str(transition.action) + " "

            if transition.link.childNode.alignedNode is not None:
                mainStr += "*"
        
        if not test:
            tmp = np.random.rand(1)
            if tmp > 0.5:
                corpus = qnA.corpus
            else:
                corpus = qnB.corpus
            corpus.AddTransition(transition)

        if transition.nextCandidates.Count() == 0:
            break

        if len(transition.visited) >= params.maxDocs:
            break

    if verbose > 0:
        mainStr += " " + str(len(ret)) 
        rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)
        print(actionStr)
        print(mainStr)
        print(rewardStr)

    return ret, totReward, totDiscountedReward

######################################################################################
def Train(params, sess, saver, qns, envs, envsTest):
    print("Start training")
    RunRLSavePlots(sess, qns, params, envsTest, params.saveDirPlots, 0, "test")
    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        for env in envs:
            TIMER.Start("Trajectory")
            arrRL, totReward, totDiscountedReward = Trajectory(env, params, sess, qns, False, 0)
            TIMER.Pause("Trajectory")
            print("epoch train", epoch, env.rootURL, totReward, totDiscountedReward)

            SavePlot(params, env, params.saveDirPlots, epoch, "train", arrRL, totReward, totDiscountedReward)

        TIMER.Start("Train")
        qns.q[0].corpus.Train(sess, params)
        qns.q[1].corpus.Train(sess, params)
        TIMER.Pause("Train")

        if epoch > 0 and epoch % params.walk == 0:
            print("Validating")
            #SavePlots(sess, qns, params, envs, params.saveDirPlots, epoch, "train")
            RunRLSavePlots(sess, qns, params, envsTest, params.saveDirPlots, epoch, "test")

        #params.eps *= 0.95
        #params.eps = max(params.eps, 0.1) 
        #print("eps", params.eps)

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

    if not os.path.exists("pickled_domains/"): os.makedirs("pickled_domains/", exist_ok=True)
    languages = GetLanguages(options.configFile)
    params = LearningParams(languages, options.saveDir, options.saveDirPlots, options.deleteDuplicateTransitions, options.langPair, languages.maxLangId, languages.GetLang("None"))

    if not os.path.exists(options.saveDirPlots): os.makedirs(options.saveDirPlots, exist_ok=True)

    #hosts = ["http://vade-retro.fr/"]
    hosts = ["http://www.buchmann.ch/", "http://telasmos.org/", "http://tagar.es/"]
    #hosts = ["http://tagar.es/"]
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
