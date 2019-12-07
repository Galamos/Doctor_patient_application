# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:26:52 2019

@author: Stella Galamo
"""
import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:
    
    #MAE - mean absolute error, the mean, or average absolute values of each error in rating predictions
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)
    
    #RMSE - root mean square error
    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)
    
    #GetTopN sort the recommendations according to relevance
    def GetTopN(predictions, n=10, minimumRating=4):
        topN = defaultdict(list)

        for userID, specialistID, actualRating, estimatedRating, _ in predictions:
            if(estimatedRating >= minimumRating):
                topN[int(userID)].append((int(specialistID), estimatedRating))
                
        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]
            
        return topN
    
    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0
        
        #For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutSpecialistID = leftOut[1]
            
            #Check if it is the predicted rating for the 10 for this user
            hit = False
            for specialistID, predictedRating in topNPredicted[int(userID)]:
                if(int(leftOutSpecialistID) == int(specialistID)):
                    hit = True
                    break
            if (hit):
                hits += 1
                
            total += 1
            
        #Compute the overall precision
        return hits/total
    
    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutOff=0):
        hits = 0
        total = 0
        
        #For each left-out rating
        for userID, leftOutSpecialistID, actualRating, estimatedRating, _ in leftOutPredictions:
            #rate the recommender's ability to recommend specialists that the user actually was satisfied with
            hit = False
            for specialistID, predictedRating in topNPredicted[int(userID)]:
                if(int(leftOutSpecialistID) == specialistID):
                    hit = True
                    break
                if (hit):
                    hits += 1
                    
                total += 1
                
            #compute overall precision
            return hits/total
        
        
    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutSpecialistID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for specialistID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutSpecialistID) == specialistID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])
            

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutSpecialistID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for specialistID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutSpecialistID) == specialistID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for specialistID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                specialist1 = pair[0][0]
                specialist2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(specialist1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(specialist2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                specialistID = rating[0]
                rank = rankings[specialistID]
                total += rank
                n += 1
        return total / n