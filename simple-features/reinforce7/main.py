#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import hashlib
import tensorflow as tf

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(relDir)
relDir = os.path.dirname(relDir)
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
    def __init__(self, languages, options, maxLangId, defaultLang):
        self.gamma = options.gamma
        self.lrn_rate = options.lrn_rate # 0.001
        self.alpha = 0.7
        self.max_epochs = 100001
        self.eps = 0.1
        self.maxBatchSize = 3000
        self.minCorpusSize = 3000
        self.updateFrequency = options.updateFrequency

        self.debug = False
        self.NUM_ACTIONS = 200
        self.NUM_LINK_FEATURES = 2

        self.saveDir = options.saveDir
        self.createFigure = bool(options.createFigure)
        
        self.reward = 1.0 #17.0
        self.cost = 0 #-1.0
        self.maxCrawl = options.maxCrawl

        self.maxLangId = maxLangId
        self.defaultLang = defaultLang

        langPairList = options.langPair.split(",")
        assert(len(langPairList) == 2)
        self.langIds = np.empty([1,2], dtype=np.int32)
        self.langIds[0,0] = languages.GetLang(langPairList[0])
        self.langIds[0,1] = languages.GetLang(langPairList[1])
        #print("self.langs", self.langs)

        self.hiddenDim = options.hiddenDim
        self.linkDim = options.linkDim

######################################################################################
def RunRLSavePlots(sess, createFigure, qn, corpus, params, envs, saveDir, epoch, sset):
    for env in envs:
        RunRLSavePlot(sess, createFigure, qn, corpus, params, env, saveDir, epoch, sset)

def RunRLSavePlot(sess, createFigure, qn, corpus, params, env, saveDir, epoch, sset):
    arrRL, totReward, totDiscountedReward = Trajectory(env, params, sess, qn, corpus, True)
    SavePlot(params, env, createFigure, saveDir, epoch, sset, arrRL, totReward, totDiscountedReward)

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
            numActions, numCandidates, linkSpecific = candidates.GetMask()
            self.numActions = numActions
            self.numCandidates = np.array(numCandidates, copy=True) 
            self.linkSpecific = np.array(linkSpecific, copy=True) 

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
def Trajectory(env, params, sess, qn, corpus, test):
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
        #print("candidates", transition.nextCandidates.Debug())
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
            corpus.AddTransition(transition)

        if transition.nextCandidates.Count() == 0:
            break

        if len(transition.visited) > params.maxCrawl:
            break

    if test:
        mainStr += " " + str(len(ret)) 
        rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)
        print(actionStr)
        print(mainStr)
        print(rewardStr)

    return ret, totReward, totDiscountedReward

######################################################################################
def Train(params, sess, saver, qn, corpus, envs, envsTest):
    print("Start training")
    for epoch in range(params.max_epochs):
        #print("epoch", epoch)
        for env in envs:
            TIMER.Start("Trajectory")
            arrRL, totReward, totDiscountedReward = Trajectory(env, params, sess, qn, corpus, False)
            TIMER.Pause("Trajectory")

            lastLangVisited = corpus.transitions[-1].langsVisited
            print("epoch train", epoch, env.rootURL, arrRL[-1], totReward, totDiscountedReward, lastLangVisited)

            if epoch % params.updateFrequency == 0:
                print("params.createFigure", params.createFigure)
                SavePlot(params, env, params.createFigure, params.saveDir, epoch, "train", arrRL, totReward, totDiscountedReward)

            TIMER.Start("CalcGrads")
            qn.CalcGrads(sess, corpus)
            TIMER.Pause("CalcGrads")

        if epoch % params.updateFrequency == 0:
            RunRLSavePlots(sess, params.createFigure, qn, corpus, params, envsTest, params.saveDir, epoch, "test")

            if epoch != 0:
                print("UpdateGrads & Validating")
                TIMER.Start("UpdateGrads")
                qn.UpdateGrads(sess, corpus)
                TIMER.Pause("UpdateGrads")


        sys.stdout.flush()
        
######################################################################################
def Temp():
    a = np.empty([2,3])
    a[0,0] = 1; a[0,1] = 2; a[0,2] = 3
    a[1,0] = 4; a[1,1] = 5; a[1,2] = 6
    print("a", a)
    a = np.reshape(a, [6])
    print("a", a)

    s = {}
    s[34] = 545
    s[a] = 7

    sasasdasd

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
    oparser.add_argument("--num-train-hosts", dest="numTrainHosts", type=int,
                         default=1, help="Number of domains to train on")
    oparser.add_argument("--num-test-hosts", dest="numTestHosts", type=int,
                         default=3, help="Number of domains to test on")
    oparser.add_argument("--max-crawl", dest="maxCrawl", type=int,
                         default=sys.maxsize, help="Maximum number of pages to crawl")
    oparser.add_argument("--gamma", dest="gamma", type=float,
                         default=0.999, help="Reward discount")
    oparser.add_argument("--update-freq", dest="updateFrequency", type=int,
                         default=5, help="Number of epoch between model gradient updates")
    oparser.add_argument("--learning-rate", dest="lrn_rate", type=float,
                         default=0.001, help="Model learning rate")
    oparser.add_argument("--hidden-dim", dest="hiddenDim", type=int,
                         default=10, help="Hidden dimension")
    oparser.add_argument("--link-dim", dest="linkDim", type=int,
                         default=5, help="Link dimension")
    oparser.add_argument("--create-figure", dest="createFigure", type=int,
                         default=1, help="Create figures")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)
    #Temp()

    if not os.path.exists(options.saveDir): os.makedirs(options.saveDir, exist_ok=True)
    if not os.path.exists("pickled_domains"): os.makedirs("pickled_domains", exist_ok=True)

    languages = GetLanguages(options.configFile)
    params = LearningParams(languages, options, languages.maxLangId, languages.GetLang("None"))

    print("options.numTrainHosts", options.numTrainHosts)
    #hosts = ["http://vade-retro.fr/"]
    #hosts = ["http://www.buchmann.ch/"]
    hosts = ["http://telasmos.org/", "http://tagar.es/", "http://www.buchmann.ch/"]
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
        corpus = Corpus(params, sess, qn)

        Train(params, sess, saver, qn, corpus, envs, envsTest)

######################################################################################
main()
