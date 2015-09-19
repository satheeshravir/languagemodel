'''python BetterKLDivergence.py corpus.txt test.txt'''

import sys
import nltk
import math
import copy


#Tokenize the sentences in given file path
def tokenize(path):
    #Tokenize Questions and answers
    sentencesFile = open(path)
    sentences = sentencesFile.readlines()
    #List of words in questions and answers
    tokens = []
    tokens += tokenizeSentenceList(sentences)
    sentencesFile.close()
    return tokens


#Tokenizes list of sentences
def tokenizeSentenceList(sentenceList):
    tokens = []
    for sentence in sentenceList:
        if ":::" in sentence:
            words,ans = sentence.split(":::")
            tokens += words.strip().lower().split()
        else:
            tokens += sentence.strip().lower().split()
    return tokens

#Generate background Unigram LM with the given tokens
def generateUnigramCorpus(tokens):
    unigramCorpusFD = nltk.FreqDist(tokens)
    return unigramCorpusFD
    
#Generate backgraound Bigram LM with the given tokens
def generateBigramCorpus(tokens):
    bigrams = list(nltk.bigrams(tokens))
    bigramCorpusFD = nltk.FreqDist(bigrams)
    return bigramCorpusFD
    
#Creates a LM skeleton for individual questions and answers
def generateShallowLM(entireCorpus):
    corpusLM = {}
    for ngram,count in entireCorpus.iteritems():
        corpusLM[ngram] = 0
    return corpusLM
    
#Calculate the KL divergence for given question and answers LM
def KLDivergence(overallAnswersLM,questionLM,question,backgroundLM):
    ngrams = backgroundLM.keys()
    totalVocabCount = sum(backgroundLM.values())
    minDistance = sys.maxint
    ansLineNumber = -1
    lineNumber=1
    for answerLM in overallAnswersLM:
        answerVocabCount = sum(answerLM.values())
        questionVocabCount = sum(questionLM.values())
        d = -1
        for ngram in ngrams:
            #Calculate the probability dist of specific ngram in answer sentence
            probAnsDist = float(answerLM[ngram])/answerVocabCount
            #Calculate the joint probability dist of specific ngram
            jointProb = float(backgroundLM[ngram])/totalVocabCount
            #Add smoothing to avoid zero division
            ansSmoothing = float(probAnsDist* 0.90) + float(0.10 * jointProb)
            #Calculate the probability dist of specific ngram in test sentence
            probQuesDist = float(questionLM[ngram])/questionVocabCount
            logx = float(probQuesDist)/ansSmoothing
            if probQuesDist != 0:
                calculation = probQuesDist * math.log(logx)
                #Add the corresponding ngram entropy to overall distance measure
                d = d + calculation
        if d < minDistance:
            minDistance = d
            ansLineNumber = lineNumber
        lineNumber+=1
    return ansLineNumber

#Generate Unigram and Bigram Answers upfront to avoid redundant calculations
def generateUnigramAndBigramAnswersLM(filePath,shallowBigramLM,shallowUnigramLM):
    answerFile = open(filePath)
    overallAnswersBigramLM = []
    overallAnswersUnigramLM = []
    for answer in answerFile:
        tokens = answer.strip().lower().split()
        answerBigramLM = copy.deepcopy(shallowBigramLM)
        answerUnigramLM = generateUnigramLM(tokens, shallowUnigramLM)
        answerBigramLM = generateBigramLM(tokens,shallowBigramLM)
        overallAnswersBigramLM.append(answerBigramLM)
        overallAnswersUnigramLM.append(answerUnigramLM)
    answerFile.close()
    return overallAnswersUnigramLM,overallAnswersBigramLM

#Generate Unigram LMs for tokens from shallowLM
def generateUnigramLM(tokens, shallowUnigramLM):
    answerUnigramLM = copy.deepcopy(shallowUnigramLM)
    for token in tokens:
        answerUnigramLM[token] += 1
    return answerUnigramLM

#Generate Bigram LMs for tokens from shallowLM
def generateBigramLM(tokens, shallowBigramLM):
    answerBigramLM = copy.deepcopy(shallowBigramLM)
    ansBigrams = list(nltk.bigrams(tokens))
    for token in ansBigrams:
        answerBigramLM[token] += 1
    return answerBigramLM
    

#Print Answers in a acceptable way
def printAnswers(lineNo, corpusFilePath, outputFile, ngram):
    outputFile.write(ngram+"\n")
        
    answerFile = open(corpusFilePath)
    i=1
    for line in answerFile:
        if i == ansLine:
            outputFile.write(line+"\n")
            break
        i+=1
    answerFile.close()

if __name__ == "__main__":
    debug = 1
    #Input file
    corpusFilePath = sys.argv[1]
    questionsFilePath = sys.argv[2]
    
    #Generate tokens for background LM
    questionTokens = tokenize(questionsFilePath)
    answerTokens = tokenize(corpusFilePath)
    overallTokens = questionTokens + answerTokens
    
    #Generate Unigram and bigram background LM for the tokens
    backgroundUnigramLM = generateUnigramCorpus(overallTokens)
    backgroundBigramLM = generateBigramCorpus(overallTokens)
    
    #Serves as a skeleton LM for individual answers and questions
    shallowUnigramLM = generateShallowLM(backgroundUnigramLM)
    shallowBigramLM = generateShallowLM(backgroundBigramLM)
    
    #Generate answers LM to avoid redundancy
    overallAnswersUnigramLM,overallAnswersBigramLM = generateUnigramAndBigramAnswersLM(corpusFilePath,shallowBigramLM,shallowUnigramLM)
    
    
    questionFile = open(questionsFilePath)
    #Iterate through the questions for KL calculation
    outputFile = open("updatedKLResults.txt","w")
    for line in questionFile:
        question,ans = line.split(":::")
        tokens = question.strip().lower().split()
        quesUnigramLM = generateUnigramLM(tokens,shallowUnigramLM)
        quesBigramLM = generateBigramLM(tokens,shallowBigramLM)
        ansLine = KLDivergence(overallAnswersBigramLM,quesBigramLM,question,backgroundBigramLM)
        outputFile.write("Question:"+question+"\n")
        printAnswers(ansLine, corpusFilePath, outputFile, "Bigram")
        ansLine = KLDivergence(overallAnswersUnigramLM,quesUnigramLM,question,backgroundUnigramLM)
        printAnswers(ansLine, corpusFilePath, outputFile, "Unigram")
    questionFile.close()
    outputFile.close()

    

