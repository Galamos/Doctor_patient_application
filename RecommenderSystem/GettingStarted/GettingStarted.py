# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:55:49 2019

@author: Stella Galamo
"""

from Main import Main
from surprise import SVD

def BuildAntiTestSetForUser(testSubject, trainset):
    fill = trainset.global_mean

    anti_testset = []
    
    u = trainset.to_inner_uid(str(testSubject))
    
    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    return anti_testset
      
#pick an arbitrary test subject
testSubject = 10
        
main = Main()
        
print("Loadiing ratings...")
data = main.loadData()
        
userRatings = main.getUserRatings(testSubject)
loved = []
hated = []

for ratings in userRatings:
    if (int(ratings[1] > 3)):
        loved.append(ratings)
    if (int(ratings[1] < 2)):
        hated.append(ratings)
        
print("\nUser ", testSubject, "was satisfied with these specialists:")
for ratings in loved:
    print(main.getSpecialistName(ratings[0]))
print("\n...and wasn't satisfied with the following specialists:")
for ratings in hated:
    print(main.getSpecialistName(ratings[0]))
    
print("\nBuilding recommendation model...")
trainSet = data.build_full_trainset()

algo = SVD()
algo.fit(trainSet)

print("Computing recommendations...")
testSet = BuildAntiTestSetForUser(testSubject, trainSet)
predictions = algo.test(testSet)

recommendations = []

print("\nThe recommended specialists are:")
for userID, specialistID, actualRating, estimatedRating, _ in predictions:
    intSpecialistID = int(specialistID)
    recommendations.append((intSpecialistID, estimatedRating))
    
recommendations.sort(key=lambda x: x[1], reverse=True)

for ratings in recommendations[:10]:
    print(main.getSpecialistName(ratings[0]))