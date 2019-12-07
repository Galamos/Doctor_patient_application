# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:21:26 2019

@author: Stella Galamo
"""

from Main import Main
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        
testSubject = '10'
k = 10

main = Main()
data = main.loadData()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

testUserInnerID = trainSet.to_inner_uid(testSubject)

# Get the top K specialists we rated
testUserRatings = trainSet.ur[testUserInnerID]
kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])
kNeighbors = []
for rating in testUserRatings:
    if rating[1] > 3.0:
        kNeighbors.append(rating)

# Get similar specialist to stuff the user liked (weighted by rating)
candidates = defaultdict(float)
for specialistID, rating in kNeighbors:
    similarityRow = simsMatrix[specialistID]
    for innerID, score in enumerate(similarityRow):
        candidates[innerID] += score * (rating / 5.0)
    
# Build a dictionary of stuff the user has already seen
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
