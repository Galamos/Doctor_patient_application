# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:02:40 2019

@author: Stella Galamo
"""
import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np
import pandas as pd

class Main:
    #create dictionaries to map the id from the rating dataset to the names of the specialists in the specialists dataset and vice-versa 
    specialistID_to_name = {}
    name_to_specialistID = {}
    specialistsPath = '../Data/specialistClean.csv'
    ratingsPath = '../Data/specialists_ratings.csv'
    
    #Functioad to load the specialist data and ratings data 
    def loadData(self):
        
        #look for files relative to the directory being used
        os.chdir(os.path.dirname(sys.argv[0]))
        
        ratingData = 0
        self.specialistID_to_name = {}
        self.name_to_specialistID = {}
        
        #use the reader to read the rating file and specify the rating scale
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        
        ratingData = Dataset.load_from_file(self.ratingsPath, reader=reader)
        
        #open both files as csv files
        with open(self.specialistsPath, newline='', encoding='ISO-8859-1') as csvfile:
            specialistReader = csv.reader(csvfile)
            #Skip the header line
            next(specialistReader)
            for row in specialistReader:
                specialistID = float(row[1])#2nd row because the first row is the count from the df count which was exported as a csv file
                specialistName = row[2]
                self.specialistID_to_name[specialistID] = specialistName
                self.name_to_specialistID[specialistName] = specialistID
        return ratingData
    
    #Function to optain user ratings from the loaded csv files
    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if(user == userID):
                    specialistID = int(row[1])
                    rating = int(row[2])
                    userRatings.append((specialistID, rating))
                    hitUser = True
                if(hitUser and (user != userID)):
                    break
        
        return userRatings
    
    #Function to compute the rated specialists popularity ranks
    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                specialistID = int(row[1])
                ratings[specialistID] += 1
        rank = 1
        for specialistID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[specialistID] = rank
            rank += 1
        return rankings
    
    #Function to get the attributes(demographic details) of specialists
    def getAttributes(self):
        attributes = defaultdict(list)
        attributeIDs = {}
        maxAttributeID = 0
        with open(self.specialistsPath, newline='', encoding='ISO-8859-1') as csvfile:
            specialistReader= csv.reader(csvfile)
            #skip header line
            next(specialistReader)
            
            for row in specialistReader:
                specialistID = float(row[1])
                attributeList = row[3].split('|')
                attributeIDList = []
                
                for attribute in attributeList:
                    if attribute in attributeIDs:
                        attributeID = attributeIDs[attribute]
                    else:
                        attributeID = maxAttributeID
                        attributeIDs[attribute] = attributeID
                        maxAttributeID += 1
                    attributeIDList.append(attributeID)
                attributes[specialistID] = attributeIDList
                
        #convert integer-encoded attribute list to bitfielfs that can be treated as vectors
        for(specialistID, attributeIDList) in attributes.items():
            bitfield = [0] * maxAttributeID
            for attributeID in attributeIDList:
                bitfield[attributeID] = 1
            attributes[specialistID] = bitfield
            
        return attributes
    
    def getSpecialistName(self, specialistID):
        if specialistID in self.specialistID_to_name:
            return self.specialistID_to_name[specialistID]
        else:
            return ""
        
    def getSpecialistID(self, specialistName):
        if specialistName in self.name_to_specialistID:
            return self.name_to_specialistID[specialistName]
        else:
            return 0
                        
        
    