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
from neural_net import Qnetwork, NeuralWalk, GetNextState
from candidate import Candidates, GetLangsVisited
from save_plot import SavePlot

######################################################################################
class LearningParams:
    def __init__(self, languages, saveDir, saveDirPlots, langPair, maxCrawl, maxLangId, defaultLang):
        self.gamma = 1.0 #0.999
        self.lrn_rate = 0.001
        self.alpha = 0.7
        self.max_epochs = 100001
        self.eps = 0.1
        self.maxBatchSize = 128
        self.minCorpusSize = 200
        self.overSampling = 1
        
        self.debug = False
        self.walk = 10
        self.NUM_ACTIONS = 30
        self.FEATURES_PER_ACTION = 1

        self.saveDir = saveDir
        self.saveDirPlots = saveDirPlots
        
        self.reward = 1.0 #17.0
        self.cost = 0 #-1.0
        self.unusedActionCost = 0.0 #-555.0
        self.maxDocs = maxCrawl #500 #9999999999

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
def RunRLSavePlots(sess, qns, params, envs, saveDirPlots, epoch, sset):
    for env in envs:
        RunRLSavePlot(sess, qns, params, env, saveDirPlots, epoch, sset)

def RunRLSavePlot(sess, qn, params, env, saveDirPlots, epoch, sset):
    arrRL, totReward, totDiscountedReward = Trajectory(env, params, sess, qn, True)
    SavePlot(params, env, saveDirPlots, epoch, sset, arrRL, totReward, totDiscountedReward)

######################################################################################
class Transition:
    def __init__(self, env, action, reward, link, langIds, visited, candidates, nextVisited, nextCandidates):
        self.action = action
        self.link = link

        self.langIds = langIds 
        self.reward = reward
        self.discountedReward = None

        if visited is not None:
            self.visited = visited
            self.langsVisited = GetLangsVisited(visited, langIds, env)

        if candidates is not None:
            self.candidates = candidates
            numActions, parentLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings, parentMatched, linkLang = candidates.GetFeatures()
            self.numActions = numActions
            self.parentLang = np.array(parentLang, copy=True) 
            self.mask = np.array(mask, copy=True) 
            self.numSiblings = np.array(numSiblings, copy=True) 
            self.numVisitedSiblings = np.array(numVisitedSiblings, copy=True) 
            self.numMatchedSiblings = np.array(numMatchedSiblings, copy=True) 
            self.parentMatched = np.array(parentMatched, copy=True) 
            self.linkLang = np.array(linkLang, copy=True) 

        self.nextVisited = nextVisited
        self.nextCandidates = nextCandidates
    
    def Debug(self):
        ret = str(self.link.parentNode.urlId) + "->" + str(self.link.childNode.urlId) + " " \
            + str(len(self.visited)) + " " + str(self.candidates.Count()) + " " \
            + str(self.reward) + " " + str(self.discountedReward)

        return ret
    
######################################################################################
def Neural(env, params, prevTransition, sess, qn):
    nextCandidates = prevTransition.nextCandidates.copy()
    nextVisited = prevTransition.nextVisited.copy()

    action, link, reward = NeuralWalk(env, params, params.eps, nextCandidates, nextVisited, sess, qn)
    assert(link is not None)
    #print("qValues", qValues.shape, action, prevTransition.nextCandidates.Count(), nextCandidates.Count())
    nextCandidates.Group(nextVisited)

    transition = Transition(env,
                            action, 
                            reward,
                            link,
                            params.langIds,
                            prevTransition.nextVisited,
                            prevTransition.nextCandidates,
                            nextVisited,
                            nextCandidates)

    return transition, reward

######################################################################################
def Trajectory(env, params, sess, qn, test):
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

    transition = Transition(env, -1, 0, None, params.langIds, None, None, nextVisited, nextCandidates)
    #print("candidates", transition.nextCandidates.Debug())

    if test:
        mainStr = "lang:" + str(startNode.lang)
        rewardStr = "rewards:"
        actionStr = "actions:"

    while True:
        transition, reward = Neural(env, params, transition, sess, qn)
        #print("visited", len(transition.visited))
        #print("candidates", transition.nextCandidates.Debug())
        #print("transition", transition.Debug())
        #print()

        numParallelDocs = NumParallelDocs(env, transition.visited)
        ret.append(numParallelDocs)

        totReward += reward
        totDiscountedReward += discount * reward
        discount *= params.gamma

        if test:
            mainStr += "->" + str(transition.link.childNode.lang)
            rewardStr += "->" + str(reward)
            actionStr += str(transition.action) + " "

            if transition.link.childNode.alignedNode is not None:
                mainStr += "*"
        else:
            qn.corpus.AddTransition(transition)

        if transition.nextCandidates.Count() == 0:
            break

        if len(transition.visited) > params.maxDocs:
            break

    if test:
        mainStr += " " + str(len(ret)) 
        rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)
        print(actionStr)
        print(mainStr)
        print(rewardStr)

    return ret, totReward, totDiscountedReward

######################################################################################
def Train(params, sess, saver, qn, envs, envsTest):
    print("Start training")
    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        for env in envs:
            TIMER.Start("Trajectory")
            arrRL, totReward, totDiscountedReward = Trajectory(env, params, sess, qn, False)
            TIMER.Pause("Trajectory")
            print("epoch train", epoch, env.rootURL, totReward, totDiscountedReward)

            #if epoch > 0 and epoch % params.walk == 0:
            SavePlot(params, env, params.saveDirPlots, epoch, "train", arrRL, totReward, totDiscountedReward)

        TIMER.Start("Train")
        qn.corpus.Train(sess, params)
        TIMER.Pause("Train")

        if epoch > 0 and epoch % params.walk == 0:
            print("Validating")
            #SavePlots(sess, qn, params, envs, params.saveDirPlots, epoch, "train")
            RunRLSavePlots(sess, qn, params, envsTest, params.saveDirPlots, epoch, "test")

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
    oparser.add_argument("--num-train-hosts", dest="numTrainHosts", type=int,
                         default=1, help="Number of domains to train on")
    oparser.add_argument("--num-test-hosts", dest="numTestHosts", type=int,
                         default=3, help="Number of domains to test on")
    oparser.add_argument("--max-crawl", dest="maxCrawl", type=int,
                         default=sys.maxsize, help="Maximum number of pages to crawl")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    languages = GetLanguages(options.configFile)
    params = LearningParams(languages, options.saveDir, options.saveDirPlots, options.langPair, options.maxCrawl, languages.maxLangId, languages.GetLang("None"))

    if not os.path.exists(options.saveDirPlots): os.makedirs(options.saveDirPlots, exist_ok=True)

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
    qn = Qnetwork(params)
    init = tf.global_variables_initializer()

    saver = None #tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        Train(params, sess, saver, qn, envs, envsTest)

######################################################################################
main()
