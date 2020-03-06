import os
import sys
import numpy as np
from tldextract import extract
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pylab as plt

from other_strategies import dumb, randomCrawl, balanced
from candidate import Candidates, UpdateLangsVisited
from helpers import NumParallelDocs
from neural_net import NeuralWalk, GetNextState

######################################################################################
def Walk(env, params, sess, qns):
    ret = []
    visited = set()
    langsVisited = np.zeros([1, 3]) # langId -> count
    candidates = Candidates(params, env)
    node = env.nodes[sys.maxsize]

    #stopNode = env.nodes[0]
    #link = Link("", 0, stopNode, stopNode)
    #candidates.AddLink(link)

    mainStr = "lang:" + str(node.lang)
    rewardStr = "rewards:"
    actionStr = "actions:"

    i = 0
    numAligned = 0
    totReward = 0.0
    totDiscountedReward = 0.0
    discount = 1.0

    while True:
        qnA = qns.q[0]
        assert(node.urlId not in visited)
        #print("node", node.Debug())
        visited.add(node.urlId)
        #print("node.lang", node.lang, langsVisited.shape)
        UpdateLangsVisited(langsVisited, node, params.langIds)        
        #print("   langsVisited", langsVisited)

        candidates.AddLinks(node, visited, params)

        numParallelDocs = NumParallelDocs(env, visited)
        ret.append(numParallelDocs)

        #print("candidates", candidates.Debug())
        _, _, _, _, _, _, _, _, action, link, reward = NeuralWalk(env, params, 0.0, candidates, visited, langsVisited, sess, qnA)
        node = link.childNode
        #print("action", action, qValues)
        actionStr += str(action) + " "

        totReward += reward
        totDiscountedReward += discount * reward

        mainStr += "->" + str(node.lang)
        rewardStr += "->" + str(reward)

        if node.alignedNode is not None:
            mainStr += "*"
            numAligned += 1

        discount *= params.gamma
        i += 1

        if node.urlId == 0:
            break

        if len(visited) > params.maxDocs:
            break

    mainStr += " " + str(i) 
    rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)

    print(actionStr)
    print(mainStr)
    print(rewardStr)
    return ret, totReward, totDiscountedReward

######################################################################################
def SavePlots(sess, qns, params, envs, saveDirPlots, epoch, sset):
    for env in envs:
        SavePlot(sess, qns, params, env, saveDirPlots, epoch, sset)

######################################################################################
def SavePlot(sess, qns, params, env, saveDirPlots, epoch, sset):
    print("Walking", env.rootURL)
    arrDumb = dumb(env, len(env.nodes), params)
    arrRandom = randomCrawl(env, len(env.nodes), params)
    arrBalanced = balanced(env, len(env.nodes), params)
    arrRL, totReward, totDiscountedReward = Walk(env, params, sess, qns)

    url = env.rootURL
    domain = extract(url).domain

    warmUp = 200
    avgRandom = avgBalanced = avgRL = 0.0
    for i in range(len(arrDumb)):
        if i > warmUp and arrDumb[i] > 0:
            avgRandom += arrRandom[i] / arrDumb[i]
            avgBalanced += arrBalanced[i] / arrDumb[i]
            avgRL += arrRL[i] / arrDumb[i]

    avgRandom = avgRandom / (len(arrDumb) - warmUp)
    avgBalanced = avgBalanced / (len(arrDumb) - warmUp)
    avgRL = avgRL / (len(arrDumb) - warmUp)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(arrDumb, label="dumb ", color='maroon')
    ax.plot(arrRandom, label="random {0:.1f}".format(avgRandom), color='firebrick')
    ax.plot(arrBalanced, label="balanced {0:.1f}".format(avgBalanced), color='red')
    ax.plot(arrRL, label="RL {0:.1f} {1:.1f}".format(avgRL, totDiscountedReward), color='salmon')

    ax.legend(loc='upper left')
    plt.xlabel('#crawled')
    plt.ylabel('#found')
    plt.title("{sset} {domain}".format(sset=sset, domain=domain))

    fig.savefig("{dir}/{sset}-{domain}-{epoch}.png".format(dir=saveDirPlots, sset=sset, domain=domain, epoch=epoch))
    fig.show()

