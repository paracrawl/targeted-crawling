#!/usr/bin/env python3
import numpy as np
import pycountry
import requests
import argparse
import hashlib
import time
import re

from bs4 import BeautifulSoup
from common import MySQL
from helpers import Env

# Heuristics to judge whether a document if parallel.
NUM_MATCHING_IMAGES = 3


def getDocumentLanguage(url):
    """
    Splits a URL to identify if it contains a language code or not.
    If a language code is not found, it make a request to the URL
    and tries to identify the language based on the text.
    """

    # Regex to extract the country code.
    # TODO: refine regex.
    regex = r'^.{8}[^\/]*\/([a-z]{2})'

    try:
        country_code = re.findall(regex, url)[0]
        lang = pycountry.languages.get(alpha_2=country_code)
        if lang:
            return lang.name
    except:
        # Send a GET request to the URL
        from langdetect import detect
        r = requests.get(url)
        text = BeautifulSoup(r.text).title.text
        try:
            return detect(text)
        except:
            return None

def getDocumentTitle(bsoup):
    return bsoup.title.text

def getDocumentImages(bsoup):
    images = bsoup.find_all('img')
    image_names = []
    for image in images:
        name = image['src'].split('/')[-1]
        image_names.append(name)
    return image_names

def getDocumentName(url):
    doc_token = url.split('/')[-1]
    if '_' in doc_token:
        # Return all except for last
        doc_name = doc_token.split('_')[:-1]
        return '_'.join(doc_name)
    else:
        return doc_token.split('.')[0]


# def merge_root_nodes(parallel_docs):
#     """
#     Search for keys with no child elements and insert similar nodes
#     (with same document names) as their child.
#     """
#     root_nodes = parallel_docs.keys()
#     print(root_nodes)
#     root_nodes_doc_names = [getDocumentName(root_node) for \
#                             root_node in root_nodes]

#     print(root_nodes_doc_names)
#     for doc_id, doc_name in root_nodes, root_nodes_doc_names):
#         indexes = np.where(root_nodes_doc_names == doc_name)[0]
#         print(indexes)
#         # print(doc_name[indexes])


def crawl_method1(sqlconn, env, lang='en'):
    """
    Method 1 uses the URL IDs to find parallel documents and uses
    the root node URL ID as the dict key. It add parallel documents based
    on the following criterion:
    1. If root and child node has the same document name
    2. If root and child are not the same URL
    """
    todo, parallel_docs = [], {}
    todo.append(env.rootNode)

    visited = set()

    while len(todo) > 0:
        node = todo.pop()

        if node.urlId not in visited:
            visited.add(node.urlId)
            parallel_docs[node.url] = []

            for link in node.links:
                childNode = link.childNode

                if (getDocumentName(node.url) == \
                   getDocumentName(childNode.url)) and \
                   (childNode.urlId != node.urlId):
                    parallel_docs[node.url].append(childNode.urlId)

                todo.append(childNode)

    # merge_root_nodes(parallel_docs)
    for parent_node, child_nodes in parallel_docs.items():
        print('parallel_docs root: ', parent_node)
        for child_node_id in child_nodes:
            print('    parallel_docs children: ', env.UrlId2Url(sqlconn, child_node_id))


def crawl_method2(sqlconn, env, lang='en'):
    """
    Method 2 uses the URL string to find parallel documents based on the following criterion:
    1. If root and child node has the same document name
    2. If root and child are not the same URL
    3. If the child URL is not already inseted in the dict of parallel documents
    the root node URL ID as the dict key.
    """
    todo, parallel_docs = [], {}
    todo.append(env.rootNode)

    visited = set()

    response_dict = {}
    while len(todo) > 0:
        node = todo.pop()

        if node.urlId not in visited:
            visited.add(node.urlId)
            doc_name = getDocumentName(node.url)
            if doc_name not in parallel_docs:
                parallel_docs[doc_name] = [node.url]
            else:
                if node.url not in parallel_docs[doc_name]:
                    parallel_docs[doc_name].append(node.url)

            for link in node.links:
                childNode = link.childNode

                if node.urlId not in response_dict:
                    r = requests.get(node.url)
                    response_dict[node.urlId] = BeautifulSoup(r.text, features="lxml")

                if childNode.urlId not in response_dict:
                    r = requests.get(childNode.url)
                    response_dict[childNode.urlId] = BeautifulSoup(r.text, features="lxml")

                matching_images = list(set(getDocumentImages(response_dict[node.urlId])) & \
                                       set(getDocumentImages(response_dict[childNode.urlId])))

                if ((getDocumentName(node.url) == \
                   getDocumentName(childNode.url)) and \
                   (childNode.urlId != node.urlId) and \
                   (childNode.url not in parallel_docs[getDocumentName(node.url)])) and \
                   len(matching_images) >= NUM_MATCHING_IMAGES:
                    parallel_docs[getDocumentName(node.url)].append(childNode.url)
                else:
                    todo.append(childNode)

    for parent_node, child_nodes in parallel_docs.items():
        print('parallel_docs root: ', parent_node)
        for child_node in child_nodes:
            print('    parallel_docs children: ', child_node)

    numDocPairs = NumParallelDocs(env, visited)
    print("visited", len(visited), "pairs found", numDocPairs)

def NumParallelDocs(env, visited):
    ret = 0;
    for urlId in visited:
        node = env.nodes[urlId]
        #print("node", node.Debug())

        if node.alignedNode is not None and node.alignedNode.urlId in visited:
            ret += 1

    return ret

def main():
    oparser = argparse.ArgumentParser(description="intelligent crawling with q-learning")
    oparser.add_argument("--config-file", dest="configFile", required=True,
                         help="Path to config file (containing MySQL login etc.)")
    options = oparser.parse_args()

    np.random.seed()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)}, linewidth=666)

    sqlconn = MySQL(options.configFile)

    hostName = "http://vade-retro.fr/"
    # hostName = "http://www.buchmann.ch/"

    start = time.time()
    env = Env(sqlconn, hostName)
    end = time.time()

    print('Time to build the graph took: ', end - start, ' seconds.')
    # crawl_method1(sqlconn, env)

    start = time.time()
    crawl_method2(sqlconn, env)
    end = time.time()

    print('Time to crawl the graph took: ', end - start, ' seconds.')

main()
