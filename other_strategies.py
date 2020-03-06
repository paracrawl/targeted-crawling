import os
import sys
import numpy as np

relDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#print("relDir", relDir)
sys.path.append(relDir)
from helpers import NumParallelDocs

######################################################################################
def byCrawlDate(env, maxDocs, params):
    nodes = list(env.nodes.values())
    print("nodes", len(nodes))
    nodes.sort(key=lambda x: x.crawlDate)

    ret = []
    visited = set()

    for node in nodes:
        if len(visited) >= maxDocs:
            break
        if node.urlId in (0, sys.maxsize):
            continue

        #print("   node", node.crawlDate, type(node.crawlDate))
        if node.urlId not in visited:
            visited.add(node.urlId)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

    return ret

######################################################################################
def dumb(env, maxDocs, params, breadthOrDepth):
    ret = []
    todo = []
    todo.append(env.rootNode)

    visited = set()
    langsVisited = {}

    while len(todo) > 0 and len(visited) < maxDocs:
        if breadthOrDepth == 0:
            node = todo.pop(0)
        else:
            node = todo.pop(-1)
        #print("node", node.Debug())
        
        if node.urlId not in visited:
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)

            for link in node.links:
                childNode = link.childNode
                #print("   ", childNode.Debug())
                todo.append(childNode)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

    return ret

######################################################################################
def randomCrawl(env, maxDocs, params):
    ret = []
    todo = []
    todo.append(env.rootNode)

    visited = set()
    langsVisited = {}

    while len(todo) > 0 and len(visited) < maxDocs:
        idx = np.random.randint(0, len(todo))
        node = todo.pop(idx)
        #print("node", node.Debug())
        
        if node.urlId not in visited:
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)

            for link in node.links:
                childNode = link.childNode
                #print("   ", childNode.Debug())
                todo.append(childNode)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

    return ret

######################################################################################
def balanced(env, maxDocs, params):
    ret = []
    visited = set()
    langsVisited = {}
    langsTodo = {}

    startNode = env.nodes[sys.maxsize]
    #print("startNode", startNode.Debug())
    assert(len(startNode.links) == 1)
    link = next(iter(startNode.links))

    while link is not None and len(visited) < maxDocs:
        node = link.childNode
        if node.urlId not in visited:
            #print("node", node.Debug())
            visited.add(node.urlId)
            if node.lang not in langsVisited:
                langsVisited[node.lang] = 0
            langsVisited[node.lang] += 1
            if params.debug and len(visited) % 40 == 0:
                print("   langsVisited", langsVisited)
    
            for link in node.links:
                #print("   ", childNode.Debug())
                AddTodo(langsTodo, visited, link)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

        link = PopLink(langsTodo, langsVisited, params)

    return ret

def PopLink(langsTodo, langsVisited, params):
    sum = 0
    # any nodes left to do
    for links in langsTodo.values():
        sum += len(links)
    if sum == 0:
        return None
    del sum

    # sum of all nodes visited
    sumAll = 0
    sumRequired = 0
    for lang, count in langsVisited.items():
        sumAll += count
        if lang in params.langIds:
            sumRequired += count
    sumRequired += 0.001 #1
    #print("langsVisited", sumAll, sumRequired, langsVisited)

    probs = {}
    for i in range(params.langIds.shape[1]):
        lang = params.langIds[0, i]
        #print("lang", lang)
        if lang in langsVisited:
            count = langsVisited[lang]
        else:
            count = 0
        #print("langsTodo", lang, nodes)
        prob = 1.0 - float(count) / float(sumRequired)
        probs[lang] = prob
    #print("   probs", probs)

    links = None
    rnd = np.random.rand(1)
    #print("rnd", rnd, len(probs))
    cumm = 0.0
    for lang, prob in probs.items():
        cumm += prob
        #print("prob", prob, cumm)
        if cumm > rnd[0]:
            if lang in langsTodo:
                links = langsTodo[lang]
            break
    
    if links is not None and len(links) > 0:
        link = links.pop(0)
    else:
        link = RandomLink(langsTodo)
    #print("   node", node.Debug())
    return link

def RandomLink(langsTodo):
    while True:
        idx = np.random.randint(0, len(langsTodo))
        langs = list(langsTodo.keys())
        lang = langs[idx]
        links = langsTodo[lang]
        #print("idx", idx, len(nodes))
        if len(links) > 0:
            return links.pop(0)
    raise Exception("shouldn't be here")

def AddTodo(langsTodo, visited, link):
    childNode = link.childNode
    
    if childNode.urlId in visited:
        return

    parentNode = link.parentNode
    parentLang = parentNode.lang

    if parentLang not in langsTodo:
        langsTodo[parentLang] = []
    langsTodo[parentLang].append(link)

######################################################################################
def linkText(env, maxDocs, params):
    ret = []
    visited = set()
    langsTodo = {}

    startNode = env.nodes[sys.maxsize]
    #print("startNode", startNode.Debug())
    assert(len(startNode.links) == 1)
    link = next(iter(startNode.links))

    while link is not None and len(visited) < maxDocs:
        node = link.childNode
        if node.urlId not in visited:
            #print("node", node.Debug())
            visited.add(node.urlId)
    
            for link in node.links:
                #print("   ", childNode.Debug())
                AddTodoLinkText(langsTodo, visited, link)

            numParallelDocs = NumParallelDocs(env, visited)
            ret.append(numParallelDocs)

        link = PopLinkLinkText(langsTodo, params)

    return ret

def PopLinkLinkText(langsTodo, params):
    numLang = 0
    langToPop = "don't care"
    if "fr" in langsTodo:
        numLang += 1
        langToPop = "fr"
    if "en" in langsTodo:
        numLang += 1
        langToPop = "en"

    if numLang == 2:
        rnd = np.random.rand(1)
        if rnd < 0.5:
            langToPop = "fr"
        else:
            langToPop = "en"
    
    if langToPop == "don't care":
        link = RandomLink(langsTodo)
    else:
        links = langsTodo[langToPop]
        if links is not None and len(links) > 0:
            link = links.pop(0)
        else:
            link = RandomLink(langsTodo)
        #print("   node", node.Debug())
    return link

def AddTodoLinkText(langsTodo, visited, link):
    childNode = link.childNode
    
    if childNode.urlId in visited:
        return

    linkLang = GroupLink(link)
    if linkLang not in langsTodo:
        langsTodo[linkLang] = []
    langsTodo[linkLang].append(link)

def GroupLink(link):
    #ret = GroupLang(link.parentNode.lang, langIds)
    #return ret

    #print("link.text", link.text, link.textLang)
    if link.text is None:
        return 0
    elif link.text.lower() in ['fr', 'francais', 'français']:
        ret = "fr"
    elif link.text.lower() in ['en', 'english']:
        ret = "en"
    else: # text is something else
        ret = "don't care"

    #print("link.text", ret, link.text, link.textLang, link.parentNode.url, link.childNode.url)
    #print("   ", ret)
    return ret
