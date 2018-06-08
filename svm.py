#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:12:30 2018

@author: Brent Mcmanus
"""
import pandas as pd

if __name__ == "__main__":
    trainSet = pd.read_csv('train.csv')
    testSet = pd.read_csv('test.csv')
    
    #replace male/female with a binary classification
    trainSet['Sex'] = trainSet['Sex'].map({'female': 1, 'male': 0}).astype(int)
    testSet['Sex'] = testSet['Sex'].map({'female': 1, 'male': 0}).astype(int)
    trainSet['Embarked'] = trainSet['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}
        ).astype(int)
    
    #grab labels and drop from the features
    trainTarg = trainSet["Survived"]
    trainSet = trainSet.drop("Survived", axis=1)
    print(trainSet.head())
    