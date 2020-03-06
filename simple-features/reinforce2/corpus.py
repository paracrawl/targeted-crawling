import os
import sys
import numpy as np

######################################################################################
class Corpus:
    def __init__(self, params, sess, qn):
        self.params = params
        self.transitions = []
        self.losses = []
        self.gradBuffer = qn.GetGradBuffer(sess)

    def AddTransition(self, transition):    
        self.transitions.append(transition)
