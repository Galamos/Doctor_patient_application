# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:55:39 2019

@author: Stella Galamo
"""

from Main import Main
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadSpecialistData():
    main = Main()
    print("Loading ratings...")
    data = main.loadData()
    print("\nComputing specialists popularity ranks so we can measure novelty later...")
    rankings = main.getPopularityRanks
    return (main, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(main, evaluationData, rankings) = LoadSpecialistData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")

# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(main)