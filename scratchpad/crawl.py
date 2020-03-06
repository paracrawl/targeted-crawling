#!/usr/bin/env python3
import argparse
import base64
import datetime
import os
import sys
import urllib

import requests
import tldextract
from bs4 import BeautifulSoup


######################################################################################
def strip_scheme(url):
    parsed = urllib.parse.urlparse(url)
    scheme = "%s://" % parsed.scheme
    return parsed.geturl().replace(scheme, '', 1)


def NormalizeURL(url):
    ind = url.find("#")
    if ind >= 0:
        url = url[:ind]
        # print("pageURL", pageURL)
    if url[-1:] == "/":
        url = url[:-1]

    url = strip_scheme(url)

    return url


######################################################################################
class CrawlHost:
    def __init__(self, url, outDir, maxCount):
        self.url = url
        self.outDir = outDir
        self.maxCount = maxCount
        self.count = 0
        self.visited = set()

        self.domain = tldextract.extract(url).registered_domain
        # print("self.domain", self.domain)

        if os.path.exists(self.outDir):
            if not os.path.isdir(self.outDir):
                sys.stderr.write("Must be a directory: " + self.outDir)
        else:
            os.mkdir(self.outDir)

        self.journal = open(outDir + "/journal", "w")

    def __del__(self):
        print(self.visited)

    ######################################################################################
    def Start(self):
        self.Download("START", self.url, None, None)

    ######################################################################################
    def Download(self, parentURL, url, linkStr, imgURL):
        if self.count >= self.maxCount:
            return False

        domain = tldextract.extract(url).registered_domain
        if domain != self.domain:
            return True

        normURL = NormalizeURL(url)
        if normURL in self.visited:
            return True

        self.count += 1
        self.visited.add(normURL)

        pageResponse = requests.get(url, timeout=5)

        # list of re-directions
        for histResponse in pageResponse.history:
            print("   histResponse", histResponse, histResponse.url, histResponse.headers['Content-Type'], \
                  histResponse.apparent_encoding, histResponse.encoding)
            # print(histResponse.text)

            histURL = histResponse.url
            histURL = urllib.parse.urljoin(parentURL, histURL)
            normHistURL = NormalizeURL(histURL)
            self.visited.add(normHistURL)

            self.WriteJournal(parentURL, histURL, histResponse.status_code, linkStr, imgURL)

            parentURL = histURL
            linkStr = None
            imgURL = None

        # found page, or error
        pageURL = pageResponse.url
        pageURL = urllib.parse.urljoin(parentURL, pageURL)

        print("pageResponse", pageResponse, pageURL, pageResponse.headers['Content-Type'], \
              pageResponse.apparent_encoding, pageResponse.encoding)
        # print(pageResponse.text)

        normPageURL = NormalizeURL(pageURL)
        self.visited.add(normPageURL)

        self.WriteJournal(parentURL, pageURL, pageResponse.status_code, linkStr, imgURL)

        if pageResponse.status_code == 200:
            # print("HH1", pageResponse.headers['Content-Type'])
            if pageResponse.headers['Content-Type'].find("text/html") >= 0:
                # print("HH2")
                with open(self.outDir + "/" + str(self.count) + ".html", "wb") as f:
                    f.write(pageResponse.content)

                soup = BeautifulSoup(pageResponse.content, features='html5lib')  # lxml html.parser
                # soup = BeautifulSoup(pageResponse.text, features='html5lib') # lxml html.parser

                plainText = soup.get_text()
                with open(self.outDir + "/" + str(self.count) + ".text", "w") as f:
                    f.write(plainText)

                cont = self.FollowLinks(soup, pageURL)
                return cont
            else:
                # print("HH3")
                return True
        else:
            return True

    ######################################################################################
    def FollowLinks(self, soup, pageURL):
        coll = soup.findAll('a')

        for link in coll:
            url = link.get('href')
            if url is None:
                continue
            url = url.strip()
            url = urllib.parse.urljoin(pageURL, url)

            linkStr = link.string
            # if linkStr is not None: linkStr = linkStr.strip()
            # print("url", linkStr, url)

            imgURL = link.find('img')
            if imgURL:
                # print("imgURL", imgURL, pageURL)
                imgURL = imgURL.get('src')
                if imgURL is not None:
                    imgURL = urllib.parse.urljoin(pageURL, imgURL)
                    imgURL = str(imgURL)
                    # print("   imgURL", imgURL, pageURL)
            else:
                imgURL = None

            cont = self.Download(pageURL, url, linkStr, imgURL)
            if not cont:
                return False

        return True

    ######################################################################################
    def WriteJournal(self, parentURL, url, status_code, linkStr, imgURL):
        if linkStr == None: linkStr = ""
        if imgURL == None: imgURL = ""

        linkStrB64 = base64.b64encode(linkStr.encode()).decode()

        journalStr = str(self.count) + "\t" \
                     + parentURL + "\t" + url + "\t" \
                     + str(status_code) + "\t" \
                     + str(datetime.datetime.now()) + "\t" \
                     + linkStrB64 + "\t" \
                     + imgURL \
                     + "\n"
        self.journal.write(journalStr)


######################################################################################

def Main():
    print("Starting")
    oparser = argparse.ArgumentParser(description="hieu's crawling")
    oparser.add_argument("--url", dest="url", required=True,
                         help="Starting URL to crawl")
    oparser.add_argument("--out-dir", dest="outDir", default=".",
                         help="Directory where html, text and journal will be saved. Will create if doesn't exist")
    oparser.add_argument("--max-requests", dest="maxRequests", default=10000, type=int,
                         help="Max number of user-generated requests")
    options = oparser.parse_args()

    crawler = CrawlHost(options.url, options.outDir, options.maxRequests)
    crawler.Start()

    print("Finished")


######################################################################################

if __name__ == "__main__":
    Main()
