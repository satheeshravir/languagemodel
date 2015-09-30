# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:20:38 2015

@author: maxsteal
"""
import sys
import nltk
import copy
import math

def tokenizeSentenceLowerCase(sentence):
    tokens = [w.lower() for w in nltk.word_tokenize(sentence)]    
    return tokens


def questionsAnswersMap(corpusFilePath):
    corpusFile = open(corpusFilePath,"r")
    questionsMap = {}
    answersMap = {}
    for line in corpusFile:
        question,answer = line.split("\t")
        if question not in questionsMap:
            questionsMap[question] = [answer]        
        else:
            questionsMap[question].append(answer)
        if answer not in answersMap:
            answersMap[answer] = [question]
        else:
            answersMap[answer].append(question)
    return questionsMap,answersMap

def generateUnigramLMForString(string, shallowLM):
    unigramTokens = tokenizeSentenceLowerCase(string)
    totalTokensCount = len(unigramTokens)
    unigramFreqDist = nltk.FreqDist(unigramTokens)
    return generateLMFromFreqDist(unigramFreqDist,totalTokensCount, shallowLM)
    
def generateListOfTrainingStringsUnigramLM(totalTrainingStringsList, shallowLM):
    trainingStringLMList = []
    for string in totalTrainingStringsList:
        unigramLM = generateUnigramLMForString(string, shallowLM)
        trainingStringLMList.append(unigramLM)
    return trainingStringLMList
    
def generateOverallUnigramBackgroundLM(totalTrainingStringsList):
    tokens = []
    for string in totalTrainingStringsList:
        tokens += tokenizeSentenceLowerCase(string)
    overallFreqDist = nltk.FreqDist(tokens)
    return generateLMFreqDist(overallFreqDist,len(tokens))
    
def generateLMFreqDist(freqDist,tokensCount):
    lm = {}
    for token, count in freqDist.iteritems():
        lm[token] = float(count) / tokensCount
    return lm

def generateLMFromFreqDist(freqDist,tokensCount, shallowLM):
    lm = copy.deepcopy(shallowLM)
    for token, count in freqDist.iteritems():
        lm[token] = float(count) / tokensCount
    return lm

#Function that generates question model based on the equation 2 
def generateQuestionModel(overallNgramBackgroundLM, trainingStringsLMList, queryTokens):
    ngrams = overallNgramBackgroundLM.keys()
    questionModel = {}
    smoothingFactor = 0.90
    for ngram in ngrams:
        numerator = 0.0
        denominator = 0.0
        for trainingStringLM in trainingStringsLMList:
            probOfNgramInString = smoothingFactor * trainingStringLM[ngram] + (1-smoothingFactor) * overallNgramBackgroundLM[ngram]
            probQueryTokenInString = 1.0
            for queryToken in queryTokens:
                if queryToken not in trainingStringLM:
                    probTokenInString = 0
                else:
                    probTokenInString = trainingStringLM[queryToken]
                if queryToken not in overallNgramBackgroundLM:
                    probTokenInBackgroundLM = 0
                else:
                    probTokenInBackgroundLM = overallNgramBackgroundLM[queryToken]
                probQueryTokenInString *= smoothingFactor * probTokenInString + (1-smoothingFactor) * probTokenInBackgroundLM
            numerator += probOfNgramInString * probQueryTokenInString
            denominator += probQueryTokenInString
        if denominator != 0:
            finalProb = numerator / denominator
        else:
            finalProb = 0.0
        questionModel[ngram] = finalProb
    return questionModel

#Generate model based on equation 4
def generateUnigramsModelsFromList(stringMap,shallowLM):
    ungigramLM = {}
    for key, string in stringMap.iteritems():
        ungigramLM[key] = generateUnigramLMForString(string, shallowLM)
    return ungigramLM
                
def generateShallowLM(entireCorpus):
    corpusLM = {}
    for ngram,count in entireCorpus.iteritems():
        corpusLM[ngram] = 0
    return corpusLM        

def KLDivergence(answersModel, questionModel,overallNgramBackgroundLM):
    smoothing = 0.90
    ngrams = overallNgramBackgroundLM.keys()
    minDistance = sys.maxint
    finalAnswer = ""
    for answer, answerModel in answersModel.iteritems():
        d = -1
        for ngram in ngrams:
            probAnsDist = answerModel[ngram]
            jointProbDist = overallNgramBackgroundLM[ngram]
            ansSmoothing = float(probAnsDist* smoothing) + float((1-smoothing) * jointProbDist)
            probQuesDist = questionModel[ngram]
            logx = float(probQuesDist)/ansSmoothing
            if probQuesDist != 0:
                calculation = probQuesDist * math.log(logx)
                #Add the corresponding ngram entropy to overall distance measure
                d = d + calculation
        #print d,answer
        if d < minDistance:
            minDistance = d
            finalAnswer = answer
    return finalAnswer

def generatePseudoAnswersMap(answersMap):
    pseudoAnswersMap = {}
    for answer, questionsList in answersMap.iteritems():
        pseudoAnswersMap[answer] = " ".join(questionsList)
    return pseudoAnswersMap

if __name__ == "__main__":
    debug = 1
    #Corpus File
    corpusFilePath = sys.argv[1]
    #Test File
    testFilePath = sys.argv[2]
    
    questionsMap, answersMap = questionsAnswersMap(corpusFilePath)
    totalQuestionsList = questionsMap.keys()
    totalAnswersList = answersMap.keys()
    
    #Generate pseudo answers map
    pseudoAnswersMap = generatePseudoAnswersMap(answersMap)
    #Background model generated for smoothing
    overallUnigramQuestionsBackgroundLM = generateOverallUnigramBackgroundLM(totalQuestionsList)
    #skeleton for LMs
    shallowQuestionBackgroundLM = generateShallowLM(overallUnigramQuestionsBackgroundLM)
    #Pseudo Answers LM created using equestion 4
    individualPseudoAnswersLM = generateUnigramsModelsFromList(pseudoAnswersMap, shallowQuestionBackgroundLM)
    
    
    testFile = open(testFilePath,"r")
    for line in testFile:
        #Sample Query
        query = line
        print "Question:",line
        queryUnigramTokens = tokenizeSentenceLowerCase(query)
        #Model generated using the equation 2
        questionModel = generateQuestionModel(overallUnigramQuestionsBackgroundLM,individualPseudoAnswersLM.values(),queryUnigramTokens)
        print "Answer:",KLDivergence(individualPseudoAnswersLM, questionModel, overallUnigramQuestionsBackgroundLM)    
    testFile.close()
    
    
    
    