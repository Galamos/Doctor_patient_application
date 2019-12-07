# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:24:54 2019

@author: Stella Galamo
"""

from Main import Main
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        
testSubject = '10'
k = 10

# Load our data set and compute the user similarity matrix
main = Main()
data = main.loadData()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

# Get top N similar users to our test subject
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID]

similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
kNeighbors = []
for rating in similarUsers:
    if rating[1] > 0.95:
        kNeighbors.append(rating)

# Get the specialists they rated, and add up ratings for each specialist, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
    
# Build a dictionary of specialists the user has already seen
watched = {}
for specialistID, rating in trainSet.ur[testUserInnerID]:
    watched[specialistID] = 1
    
# Get top-rated specialists from similar users:
pos = 0
for specialistID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not specialistID in watched:
        specialistID = trainSet.to_raw_iid(specialistID)
        print(main.getSpecialistName(int(specialistID)), ratingSum)
        pos += 1
        if (pos > 10):
            break