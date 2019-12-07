# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:15:40 2019

@author: Stella Galamo
"""

from Main import Main
from surprise import KNNBasic
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadSpecialistData():
    main = Main()
    print("Loading ratings...")
    data = main.loadData()
    print("\nComputing specialists popularity ranks...")
    rankings = main.getPopularityRanks
    return (main, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(main, evaluationData, rankings) = LoadSpecialistData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# User-based KNN
UserKNN = KNNBasic(sim_options = {'name': 'pearson', 'user_based': True})
evaluator.AddAlgorithm(UserKNN, "User KNN")

# Item-based KNN
ItemKNN = KNNBasic(sim_options = {'name': 'pearson', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN, "Item KNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(main)
