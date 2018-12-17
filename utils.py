# Or Zipori 302933833
# Shauli Ravfogel 308046861
import numpy as np 
from collections import Counter

vocabSize = 300000

# read file and return its lines as list
def fetchFile(filename):
    lines = []
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    return lines

# get list of lines and create a list of the words
def createWordsDataSet(linelist):
    wordsDataSet = []
    for i, line in enumerate(linelist):
        # ignore headers and blank lines
        if(i%4 == 0 or len(line) == 0):
            continue

        wordsDataSet.extend(line.split())

    return wordsDataSet

# to see that the probabilities are summing to 1
def debugModel(model):
    probs = model.probs
    sumP = 0
    for p in probs:
        sumP += probs[p]

    return sumP

# receives a list of probabilities and calculates the preplixity
def perplexity(probs):
    n = len(probs)
    sumProps = 0

    for p in probs:
        sumProps += np.log2(p) if p > 1e-20 else -np.inf
    
    return 2**(-(sumProps) / n)

# create a dictionary that contains event - > count(event)
def countEvents(wordsDataSet):
    eventCount = {}

    for w in wordsDataSet:
        if (w in eventCount):
            eventCount[w] = eventCount[w] + 1
        else:
            eventCount[w] = 1
    
    return eventCount

# Lidstone model - receives dataset(list) and lambda(scalar) as arguments
class Lidstone(object):
    def __init__(self, dataset, testset):
        self.wordsDataSet = createWordsDataSet(dataset)
        self.testset = createWordsDataSet(testset)
        # all the events    
        self.S = list(set(self.wordsDataSet))

        # split set to two sets
        divider = round(0.9 * len(self.wordsDataSet))
        self.trainset = self.wordsDataSet[:divider]
        self.validset = self.wordsDataSet[divider:]

        # event -> count(event)
        self.eventCount = None

        # probabilities
        self.probs = {}
        self.count_probs = {}
    
    # calculate the probabilities with Lidstone 
    def train(self):
           
           self.word_counter = Counter()
           for w in self.trainset:
           
           	self.word_counter[w] += 1
            
            
    def calc_probs(self, lambdaP = 0.5): 
    
        probs = {}
        
        # compute probs for trainset words
        
        for w in self.trainset:
        	
        	probs[w] = (self.word_counter[w] + lambdaP)/(len(self.trainset) + (lambdaP * vocabSize))
        	self.count_probs[self.word_counter[w]] = probs[w]
        	
        diff = vocabSize - len(self.S)
        
        for i in range(diff):
           probs["unseen" + str(i)] = lambdaP / (len(self.trainset) + (lambdaP * vocabSize))        	
        
        self.probs = probs
    
    # calculate the perplexity over the     
    def validate(self, dev = True):
        # get all the words(events) for the validation set
        validateWordsSet = self.validset if dev else self.testset
        vProbs = []

        for word in validateWordsSet:
            if (word not in self.probs):
                # gives the probability of an unseen event
                vProbs.append(self.probs["unseen0"])
            else:
                vProbs.append(self.probs[word])
        
        return perplexity(vProbs)

# Headout model
class Headout(object):
    def __init__(self, dataset, testset):
        self.wordsDataSet = createWordsDataSet(dataset)
        self.testset = createWordsDataSet(testset)
        # all the events    
        self.S = list(set(self.wordsDataSet))
        divider = round(0.5 * len(self.wordsDataSet))

        # the sets to work on
        self.trainset = self.wordsDataSet[:divider] #St
        self.headoutset = self.wordsDataSet[divider:] #Sh

        # probabilities
        self.probs = {}
        
        self.N_r = {}
        self.t_r = {}
        self.r_probs = {}

    '''
        take dictionary of event->count and return a dictionary of word buckets
        where each bucket contains words that appear r times
    '''
    def wordsToBuckets(self):
        eventCount = countEvents(self.trainset) 
        buckets = {}

        for word in eventCount:
            r = eventCount[word]
            # create a new bucket for new r
            if (r not in buckets):
                buckets[r] = [word]
            else:
                buckets[r].append(word)
        
        # for unseen words in trainset
        # case r = 0
        # get all the words that appeard in Sh but not in St
        buckets[0] = []
        diffSet = set(self.headoutset) - set(self.trainset)
        buckets[0].extend(diffSet)
        diff = vocabSize - len(diffSet) - len(set(self.trainset))
        # fill the missing unseen words
        buckets[0].extend(["unseen" + str(i) for i in range(diff)])
        
        return buckets

    def train(self):
        headoutSetCount = countEvents(self.headoutset)
        buckets = self.wordsToBuckets()

        ShSize = len(self.headoutset)
        for b in buckets:
            # get the list of words associated with r
            words = buckets[b] 
            sumCH = 0
            # count the number of occurences of words from St in Sh
            for w in words:
                if (w not in headoutSetCount):
                    sumCH += 0
                else:
                    sumCH += headoutSetCount[w]

            Nr = len(words)
            self.N_r[b] = Nr
            self.t_r[b] = sumCH
            
            # calculate the Pheadout for r=i
            pHO = sumCH / (Nr * ShSize)
            self.r_probs[b] = pHO
            
            # assign the probabilities to all the words in this bucket
            for w in words:
                self.probs[w] = pHO
                
    def get_N_r(self, r):
    	
    	return self.N_r[r] if r in self.N_r else 0
    
    def get_t_r(self, r):
    
    	return self.t_r[r] if r in self.t_r else 0
    
    def get_r_prob(self, r):
    
    	return self.r_probs[r] if r in self.r_probs else 0
    
    def validate(self, dev = True):
        # get all the words(events) for the validation set
        validateWordsSet = self.validset if dev else self.testset
        vProbs = []

        for word in validateWordsSet:
            if (word not in self.probs):
                # gives the probability of an unseen event
                vProbs.append(self.probs["unseen0"])
            else:
                vProbs.append(self.probs[word])
        
        return perplexity(vProbs)
