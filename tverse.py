#import regex
import math
import collections, itertools
import random
import re
import csv
import pprint
import nltk.classify, nltk.metrics
import nltk
from sklearn import cross_validation
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist, ConditionalFreqDist

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    #print 'process before %s||\n' % (tweet)
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')	
	#strip punctuation
    tweet = tweet.strip('\'"?,.') 
	#print 'end %s||\n' % (tweet)
    return tweet
#end 

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
	#Added 5/22/2013.  Not sure we want to do this.
    #stopWords.append('AT_USER')
    #stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()

    for w in words:
		#replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
		#strip punctuation (Moved to processTweet) 6/18/2013
		#tweet = tweet.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector    
#end

#start getFeatureList
def getFeatureList(fileName):
    fp = open(fileName, 'r')
    line = fp.readline()
    featureList = []
    while line:
        line = line.strip()
        featureList.append(line)
        line = fp.readline()
    fp.close()
    return featureList
#end

#start extract_features
def extract_features(tweet):
	tweet_words = set(tweet)
	features = {}
	for word in featureList:
		features['contains(%s)' % word] = (word in tweet_words)		
	return features
#end

#start evaluate_classifier
#texts = tweets
#featx = features
#k = 10 iterations
#fold = 0
def evaluate_classifier(texts, featx,k=10, fold=0):
    trainfeats = []
    testfeats = []
	
    # loop over tweets and assign each to one of the k folds
    for i in range(len(texts)):
        feats = featx(texts[i]['text'])
        #feats.update( { texts[i]['section'] : True } )
        label = texts[i]['label']

        #print 'idx %d, fold %d of %d' % (i, fold, k)
        if i % (k+1) == fold:
            #print 'test'
            testfeats.append((feats, label))
        else:
            #print 'train'
            trainfeats.append((feats, label))

    #print 'train set has %d tweets' % (len(trainfeats))
    #print 'test set has %d tweets' % (len(testfeats))

    classifier = nltk.NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
 
    accuracy = nltk.classify.util.accuracy(classifier, testfeats)
    print featx.__name__, 'fold', fold
    print 'accuracy:', accuracy
    print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
    print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
    print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
    print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
    
    # This can be uncommented to print informative features (uses a lot of memory)
    #classifier.show_most_informative_features(n=20)

    if fold < k:
        return [accuracy] + evaluate_classifier(texts, featx, k, fold+1)
    else:
        return [accuracy]
#end

#get the best words
def get_best_words(corpora):
    # Technically, this stuff should all be restricted to the current train set
    # so it doesn't look at labels in the text set
 
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for text in corpora['pos']:
        for word in text.tokens:
            word_fd.inc(word.lower())
            label_word_fd['pos'].inc(word.lower())
 
    for text in corpora['neg']:
        for word in text.tokens:
            word_fd.inc(word.lower())
            label_word_fd['neg'].inc(word.lower())
  
    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
 
    word_scores = {}
 
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
                                               (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
                                               (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
 
    best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
    bestwords = set([w for w, s in best])
    return bestwords
#end
	
def word_feats(featureVector):
    return dict([(word, True) for word in featureVector])

def all_word_bigram_feats(featureVector):
    bigrams = [' '.join(b) for b in nltk.util.bigrams(featureVector)]
    return dict([(word, True) for word in list(featureVector) + bigrams])

def top_bigram_word_feats(featureVector, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(featureVector)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(featureVector, bigrams)])	
	
def best_word_feats(featureVector, bestwords):
    return dict([(word, True) for word in featureVector if word in bestwords])

def best_bigram_word_feats(featureVector, bestwords, score_fn=BigramAssocMeasures.chi_sq, n=500):
    bigram_finder = BigramCollocationFinder.from_words(featureVector)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(featureVector, bestwords))
    return d	
	
#Read the tweets one by one and process it
inpTweets = csv.reader(open('tweets_training_set.csv', 'rb'), delimiter='*', quotechar='|')
st = open('stopwords.txt', 'r')
stopWords = getStopWordList('stopwords.txt')
featureList = getFeatureList('tweets_feature_list.txt')
pp = pprint.PrettyPrinter()
count = 0;
tweets = []
tweetSet = []
corpora = {}

for row in inpTweets:
    sentiment = row[0]	
    tweet = row[1]
	#print '%s %s\n' % (tweet, sentiment)
    processedTweet = processTweet(tweet)
	#break the tweet into tokens
    tokens = nltk.word_tokenize(processedTweet.lower().replace('.', ' '))
    text = nltk.Text(tokens)
    
    if sentiment not in corpora:
       corpora[sentiment] = []

    corpora[sentiment].append(text)
	
    featureVector = getFeatureVector(processedTweet, stopWords)
    for word in featureVector:
        #print '%s %s' % (word, sentiment)
        print '%s\t%s' % (word, sentiment)
		
	#print "featureVector = %s\n" % (featureVector)
    #print "tweet = %s , featureVector = %s\n" % (tweet, featureVector)
    tweets.append((featureVector, sentiment));
    tweetSet.append({'text':tweet, 'label':sentiment})
#end loop

#ensuring we have a random traning set each time
#seeding with the computers time
random.seed()
random.shuffle(tweets)

training_set = nltk.classify.util.apply_features(extract_features, tweets)

#ensuring we have a random traning set each time
#seeding with the computers time
random.seed()
random.shuffle(tweetSet)

#Test cross validation
cv = cross_validation.KFold(len(training_set), n_folds=10, indices=True, shuffle=False, random_state=None, k=None)
for traincv, testcv in cv:
    classifier = nltk.NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    print 'Overall Accuracy:', nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])


results = {}

if True:
    print 'evaluating single word features'
    accs = evaluate_classifier(tweetSet, word_feats)
    print sum(accs)/len(accs), accs
    results['words'] = accs

if True:
     print 'evaluating all word and bigram features'
     accs = evaluate_classifier(tweetSet, all_word_bigram_feats)
     print sum(accs)/len(accs), accs
     results['all_word_bigram'] = accs
	 
if True:
      print 'evaluating top bigram word features'
      accs = evaluate_classifier(tweetSet, top_bigram_word_feats)
      print sum(accs)/len(accs), accs
      results['words_top_bigrams'] = accs		
  
if True:
     print 'evaluating best word features'
     bestwords = get_best_words(corpora)
     print "best words: %s\n" % bestwords
     accs = evaluate_classifier(tweetSet, lambda words: best_word_feats(featureVector, bestwords))
     print sum(accs)/len(accs), accs
     results['best_words'] = accs
if True:
     print 'evaluating best words + bigram chi_sq word features'
     accs = evaluate_classifier(tweetSet, lambda words: best_bigram_word_feats(featureVector, bestwords))
     print sum(accs)/len(accs), accs
     results['best_words_and_top_bigrams'] = accs

print results

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

#Train the Max Entropy Classifier
MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', trace=3, \
                       encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 10)
            	
#print "NaiveBayes Training set Accuracy: %s" % (nltk.classify.accuracy(NBClassifier, training_set))
#print "Max Entropy Accuracy: %s\n" % (nltk.classify.accuracy(MaxEntClassifier, training_set))
#print "NaiveBayes Most Informative: %s\n" % (NBClassifier.show_most_informative_features(100))
#print "Max Entropy Most Informative: %s\n" % (MaxEntClassifier.show_most_informative_features(10))

#We need to read the entire set of tweets from a .csv file and run them through the classification algorithm
#Read the tweets one by one and process it
inpTweetsContent = csv.reader(open('tweets.csv', 'rb'), delimiter='*', quotechar='|')
pp = pprint.PrettyPrinter()
tweets = []
for row in inpTweetsContent:
	tweet = row[0]
	#print 'tweet %s||\n' % (tweet)
	tweetParsed = processTweet(tweet)
	sentimentNb = NBClassifier.classify(extract_features(getFeatureVector(processTweet(tweet), stopWords)))
	sentimentMe = MaxEntClassifier.classify(extract_features(getFeatureVector(processTweet(tweet), stopWords)))
	print "%s\t%s\t%s\t%s" % (tweet, tweetParsed, sentimentNb, sentimentMe)
#end loop

'''
# Test the classifier
testTweet = 'thanks for all your support'
processedTestTweet = processTweet(testTweet)
sentimentNb = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
sentimentMe = MaxEntClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
print "NaiveBayes: testTweet = %s, sentiment = %s" % (testTweet, sentimentNb)
print "MaxEntropy: testTweet = %s, sentiment = %s\n" % (testTweet, sentimentMe)
'''