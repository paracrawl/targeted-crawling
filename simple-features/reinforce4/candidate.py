import os
import sys
import numpy as np

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from common import GetLanguages, Languages, Timer
from helpers import GetVistedSiblings, GetMatchedSiblings, GetNodeMatched

######################################################################################
def GetLangsVisited(visited, langIds, env):
    langsVisited = np.zeros([1, 3]) # langId -> count

    for urlId in visited:
        node = env.nodes[urlId]
        
        if node.lang == langIds[0, 0]:
            offset = 0
        elif node.lang == langIds[0, 1]:
            offset = 1
        else:
            offset = 2

        langsVisited[0, offset] += 1

        # count unmatched
        #isMatched = GetNodeMatched(node, visited)
        #if isMatched == 0:
        #    langsVisited[0, offset] += 1
            
    return langsVisited

def GroupLang(langId, langIds):
    if langId == langIds[0, 0]:
        return 0
    elif langId == langIds[0, 1]:
        return 1
    else: 
        return 2

######################################################################################
class Candidates:
    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.links = set()
        self.grouped = {} # key -> links[]

        #for langId in params.langIds:
        #    self.dict[langId] = []

    def copy(self):
        ret = Candidates(self.params, self.env)
        ret.links = self.links.copy()
        
        for key, nodes in self.grouped.items():
            ret.grouped[key] = nodes.copy()

        return ret
    
    def Group(self, visited):
        self.grouped.clear()
        
        for link in self.links:
            parentLang = GroupLang(link.parentNode.lang, self.params.langIds)
            key = (parentLang,)

            if key not in self.grouped:
                self.grouped[key] = []
            self.grouped[key].append(link)
        
    def AddLinks(self, node, visited, params):
        #print("   currNode", curr, currNode.Debug())
        newLinks = node.GetLinks(visited, params)

        for link in newLinks:
            assert(link not in self.links)
            self.links.add(link)

    def Pop(self, key):
        assert(len(self.grouped) > 0)
        links = self.grouped[key]
        assert(len(links) > 0)

        idx = np.random.randint(0, len(links))
        #idx = 0
        link = links.pop(idx)

        # remove all links going to same node
        linksCopy = self.links.copy()
        for linkCopy in linksCopy:
            if linkCopy.childNode == link.childNode:
                self.links.remove(linkCopy)

        return link

    def Count(self):
        ret = len(self.links)
        return ret

    def GetMask(self):
        #print("self", self.Debug())
        numActions = 0
        numCandidates = np.full([1, self.params.NUM_ACTIONS], False, dtype=np.float)
        parentLang = np.empty([1, self.params.NUM_ACTIONS], dtype=np.float)
        #parentLang = np.full([1, self.params.NUM_ACTIONS], fill_value=66543.44, dtype=np.float)

        for key, nodes in self.grouped.items():
            #if numActions >= self.params.NUM_ACTIONS:
            #    break
            #print("numActions", numActions)
            assert(numActions < self.params.NUM_ACTIONS)
            assert(len(nodes) > 0)

            numCandidates[0, numActions ] += len(nodes)
            parentLang[0, numActions ] = key[0]
            numActions += 1

        return numActions, numCandidates, parentLang

    def Debug(self):
        ret = str(len(self.links)) + " "
        for key in self.grouped:
            ret += str(key) + ":" + str(len(self.grouped[key])) + " "
            #links = self.dict[key]
            #for link in links:
            #    ret += str(link.parentNode.urlId) + "->" + str(link.childNode.urlId) + " "
        return ret
