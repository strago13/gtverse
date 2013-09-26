import collections
import nltk.metrics
import re
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.classify import NaiveBayesClassifier
 
 
# Process a word
def processTweet(tweet):
    coreTweet = " ".join(tweet)	
    print 'process before %s: ' % (coreTweet)
	# process the tweets
	#Convert to lower case
    coreTweet = coreTweet.lower()
	#Convert www.* or https?://* to URL
    coreTweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',coreTweet)
	#Convert @username to AT_USER
    coreTweet = re.sub('@[^\s]+','AT_USER',coreTweet)    
	#Replace #word with word
    coreTweet = re.sub(r'#([^\s]+)', r'\1', coreTweet)
	#Remove additional white spaces
    coreTweet = re.sub('[\s]+', ' ', coreTweet)
	#trim
    coreTweet = coreTweet.strip('\'"')	
	#strip punctuation
    coreTweet = coreTweet.strip('\'"?,.|') 			
    print 'after %s\n' % (coreTweet)
    return coreTweet.split(" ")
#end  
 

 
def word_feats(words):
    return dict([(word, True) for word in words])	
	
reader = CategorizedPlaintextCorpusReader('./corpus', r'.*', cat_pattern=r'(.*)_.*')
	
for i in range(len(reader.sents())):
  reader.sents()[i] = processTweet(reader.sents()[i])	
	
#print reader.sents( )[1:2]  # etc.
#print reader.fileids(categories=['neg'])

negids = reader.fileids('neg')
posids = reader.fileids('pos')
neuids = reader.fileids('neu')
 
negfeats = [(word_feats(processTweet(reader.words(fileids=[f]))), 'neg') for f in negids]
posfeats = [(word_feats(processTweet(reader.words(fileids=[f]))), 'pos') for f in posids]
#neufeats = [(word_feats(reader.words(fileids=[f])), 'neu') for f in neuids] 
 
print "pos length %s" % (len(posfeats)) 
print "neg length %s" % (len(negfeats)) 
#print "neu length %s" % (len(neufeats)) 
 
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4
#neucutoff = len(neufeats)*3/4

print "pos cutff: %s\t neg cutoff %s\t" % (poscutoff, negcutoff)

#neucutoff = len(neufeats)*3/4
 
#trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff] + neufeats[:neucutoff]
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
#testfeats = negfeats[negcutoff:] + posfeats[poscutoff:] + neufeats[neucutoff:]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
 
print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
print 'neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg'])
print 'neu precision:', nltk.metrics.precision(refsets['neu'], testsets['neu'])
print 'neu recall:', nltk.metrics.recall(refsets['neu'], testsets['neu'])
print 'neu F-measure:', nltk.metrics.f_measure(refsets['neu'], testsets['neu'])
classifier.show_most_informative_features(10)

