#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:59:37 2019

@author: fkirefu
"""

#!/usr/bin/env python3
import sys, os 
from common import MySQL, Languages, Timer
from helpers import Env, Link
from tldextract import extract
import pickle

#############################################################
def PickleDomain(url):
    print("Pickling", url)
    domain = extract(url).domain

    if not os.path.exists('pickled_domains/'+domain):
        sqlconn = MySQL('config.ini')
        
        env = Env(sqlconn, url)
        
        with open('pickled_domains/'+domain, 'wb') as f:
            pickle.dump(env, f)

    print("Done {}".format(domain))

#############################################################
allhostNames = ["http://www.buchmann.ch/",
                "http://vade-retro.fr/",
                "http://www.visitbritain.com/",
                "http://www.lespressesdureel.com/",
                "http://www.otc-cta.gc.ca/",
                "http://tagar.es/",
                "http://lacor.es/",
                "http://telasmos.org/",
                "http://www.haitilibre.com/",
                "http://legisquebec.gouv.qc.ca/",
                "http://hobby-france.com/",
                "http://www.al-fann.net/",
                "http://www.antique-prints.de/",
                "http://www.gamersyde.com/",
                "http://inter-pix.com/",
                "http://www.acklandsgrainger.com/",
                "http://www.predialparque.pt/",
                "http://carta.ro/",
                "http://www.restopages.be/",
                "http://www.burnfateasy.info/",
                "http://www.bedandbreakfast.eu/",
                "http://ghc.freeguppy.org/",
                "http://www.bachelorstudies.fr/",
                "http://chopescollection.be/",
                "http://www.lavery.ca/",
                "http://www.thecanadianencyclopedia.ca/",
                #"http://www.vistastamps.com/",
                "http://www.linker-kassel.com/",
                "http://www.enterprise.fr/"]

sys.setrecursionlimit(9999999)

if not os.path.exists("pickled_domains"): os.makedirs("pickled_domains", exist_ok=True)

if len(sys.argv) < 2:
    for url in allhostNames:
        PickleDomain(url)
else:
    url = sys.argv[1]
    PickleDomain(url)

        
