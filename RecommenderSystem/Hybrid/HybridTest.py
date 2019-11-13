# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:32:59 2019

@author: Stella Galamo
"""

from Main import Main
from RBMAlgorithm import RBMAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator

import random
import numpy as np

def LoadSpecialistData():
    main = Main()
    print("Loading ratings...")
    data = main.loadData()
    print("\nComputing specialist popularity ranks so we can measure novelty later...")
    rankings = main.getPopularityRanks()
    return (main, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(main, evaluationData, rankings) = LoadSpecialistData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#Simple RBM
SimpleRBM = RBMAlgorithm(epochs=40)
#Content
ContentKNN = ContentKNNAlgorithm()

#Combine them
Hybrid = HybridAlgorithm([SimpleRBM, ContentKNN], [0.5, 0.5])

evaluator.AddAlgorithm(SimpleRBM, "RBM")
evaluator.AddAlgorithm(ContentKNN, "ContentKNN")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(main)
