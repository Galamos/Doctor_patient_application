# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:27:14 2019

@author: Stella Galamo
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from Main import Main
import math
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # Compute specialist similarity matrix based on content attributes

        # Load up attribute vectors for every specialist
        main = Main()
        attributes = main.getAttributes()
        
        print("Computing content-based similarity matrix...")
            
        # Compute attribute distance for every specialist combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for thisRating in range(self.trainset.n_items):
            if (thisRating % 100 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisSpecialistID = int(self.trainset.to_raw_iid(thisRating))
                otherSpecialistID = int(self.trainset.to_raw_iid(otherRating))
                attributeSimilarity = self.computeAttributesSimilarity(thisSpecialistID, otherSpecialistID, attributes)
                self.similarities[thisRating, otherRating] = attributeSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
                
        print("...done.")
                
        return self
    
    def computeAttributesSimilarity(self, specialist1, specialist2, attributes):
        attributes1 = attributes[specialist1]
        attributes2 = attributes[specialist2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(attributes1)):
            x = attributes1[i]
            y = attributes2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or specialist is unkown.')
        
        # Build up similarity scores between this item and everything the sample user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            attributeSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (attributeSimilarity, rating[1]) )
        
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by sample user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
