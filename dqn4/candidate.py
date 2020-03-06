import os
import sys
import numpy as np

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from common import GetLanguages, Languages, Timer
from helpers import GetVistedSiblings, GetMatchedSiblings

######################################################################################
def UpdateLangsVisited(langsVisited, node, langIds):
        if node.lang == langIds[0, 0]:
            langsVisited[0, 0] += 1
        elif node.lang == langIds[0, 1]:
            langsVisited[0, 1] += 1
        else:
            langsVisited[0, 2] += 1

######################################################################################
class Candidates:
    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.dict = {} # key -> links[]

        #for langId in params.langIds:
        #    self.dict[langId] = []

    def copy(self):
        ret = Candidates(self.params, self.env)

        for key, value in self.dict.items():
            #print("key", key, value)
            ret.dict[key] = value.copy()

        return ret
    
    def AddLink(self, link, visited):
        langId = link.parentNode.lang
        numSiblings = len(link.parentNode.links)
        
        numVisitedSiblings = GetVistedSiblings(link.childNode.urlId, link.parentNode, visited)
        numVisitedSiblings = len(numVisitedSiblings)

        matchedSiblings = GetMatchedSiblings(link.childNode.urlId, link.parentNode, visited)
        numMatchedSiblings = len(matchedSiblings)
        
        #print("numSiblings", numSiblings, numMatchedSiblings, link.childNode.url)
        #for sibling in link.parentNode.links:
        #    print("   sibling", sibling.childNode.url)

        key = (langId,numSiblings, numVisitedSiblings, numMatchedSiblings) 
        if key not in self.dict:
            self.dict[key] = []
        self.dict[key].append(link)
        
    def AddLinks(self, node, visited, params):
        #print("   currNode", curr, currNode.Debug())
        newLinks = node.GetLinks(visited, params)

        for link in newLinks:
            self.AddLink(link, visited)

    def Pop(self, key):
        links = self.dict[key]
        assert(len(links) > 0)

        idx = np.random.randint(0, len(links))
        link = links.pop(idx)

        # remove all links going to same node
        for otherLinks in self.dict.values():
            otherLinksCopy = otherLinks.copy()
            for otherLink in otherLinksCopy:
                if otherLink.childNode == link.childNode:
                    otherLinks.remove(otherLink)

        return link

    def Count(self):
        ret = 0
        for _, dict in self.dict.items():
            ret += len(dict)
        return ret

    def GetFeatures(self):
        numActions = 0
        linkLang = np.zeros([1, self.params.MAX_NODES], dtype=np.int32)
        numSiblings = np.zeros([1, self.params.MAX_NODES], dtype=np.int32)
        numVisitedSiblings = np.zeros([1, self.params.MAX_NODES], dtype=np.int32)
        numMatchedSiblings = np.zeros([1, self.params.MAX_NODES], dtype=np.int32)

        mask = np.full([1, self.params.MAX_NODES], False, dtype=np.bool)
        
        for key, nodes in self.dict.items():
            if len(nodes) > 0:
                assert(numActions < self.params.MAX_NODES)
                linkLang[0, numActions] = key[0]
                numSiblings[0, numActions] = key[1]
                numVisitedSiblings[0, numActions] = key[2]
                numMatchedSiblings[0, numActions] = key[3]

                mask[0, numActions] = True
                numActions += 1

        return numActions, linkLang, mask, numSiblings, numVisitedSiblings, numMatchedSiblings

    def Debug(self):
        ret = ""
        for lang in self.dict:
            ret += "lang=" + str(lang) + ":" + str(len(self.dict[lang])) + " "
            #links = self.dict[lang]
            #for link in links:
            #    ret += " " + link.parentNode.url + "->" + link.childNode.url
        return ret
