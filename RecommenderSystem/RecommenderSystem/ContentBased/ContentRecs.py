# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:15:34 2019

@author: Stella Galamo
"""

from Main import Main
from ContentKNNAlgorithm import ContentKNNAlgorithm
from Evaluator import Evaluator
from surprise import NormalPredictor

import random
import numpy as np

def LoadSpecialistData():
    main = Main()
    print("Loading ratings...")
    data = main.loadData()
    print("\nComputing popularity ranks so we can measure novelty later...")
    rankings = main.getPopularityRanks()
    return (main, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(main, evaluationData, rankings) = LoadSpecialistData()

# Construct an Evaluator to evaluate them
evaluator = Evaluator(evaluationData, rankings)

contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(main)