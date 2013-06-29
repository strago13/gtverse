from nltk.probability import DictionaryProbDist
from nltk import NaiveBayesClassifier
from nltk import FreqDist, ConditionalFreqDist
from nltk import BigramAssocMeasures
 
train_samples = {
    'I hate you and you are a bad person': 'neg',
    'I love you and you are a good person': 'pos',
    'I fail at everything and I want to kill people' : 'neg',
    'I win at everything and I want to love people' : 'pos',
    'sad are things are heppening. fml' : 'neg',
    'good are things are heppening. gbu' : 'pos',
    'I am so poor' : 'neg',
    'I am so rich' : 'pos',
    'I hate you mommy ! You are my terrible person' : 'neg',
    'I love you mommy ! You are my amazing person' : 'pos',
    'I want to kill butterflies since they make me sad' : 'neg',
    'I want to chase butterflies since they make me happy' : 'pos',
    'I want to hurt bunnies' : 'neg',
    'I want to hug bunnies' : 'pos',
    'You make me frown' : 'neg',
    'You make me smile' : 'pos',
}
 
word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()
 
for words, label  in train_samples.items():
    for word in words.split():
        word_fd.inc(word.lower())
        label_word_fd[label].inc(word.lower())
 
print word_fd
print label_word_fd
 
pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count
 
print 'pos word count', pos_word_count
print 'neg word count', neg_word_count
print 'total word count', total_word_count
 
word_scores = {}
 
for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
        (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
        (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score
 
print 'word scores', word_scores
 
best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:100]
bestwords = set([w for w, s in best])
 
print '*' * 50
print 'best words', bestwords
print '*' * 50
 
def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])
 
 
test_samples = [
  'You are a terrible person and everything you do is bad',
  'I love you all and you make me happy',
  'I frown whenever I see you in a poor state of mind',
  'Finally getting rich from my ideas. They make me smile.',
  'My mommy is poor',
  'I love butterflies. Yay for happy',
  'Everything is fail today and I hate stuff',
]
 
 
def gen_bow(text):
    words = text.split()
    bow = {}
    for word in words:
        bow[word.lower()] = True
    return bow
 
label_probdist = DictionaryProbDist({'pos': 0.5, 'neg': 0.5})
 
true_probdist = DictionaryProbDist({True: 6})
 
feature_probdist = { ## need to generate this from train_samples
            ('neg', 'no'): true_probdist,
            ('neg', 'hate'): true_probdist,
            ('neg', 'fml'): true_probdist,
            ('neg', 'poor'): true_probdist,
            ('neg', 'sad'): true_probdist,
            ('neg', 'fail'): true_probdist,
            ('neg', 'kill'): true_probdist,
            ('neg', 'evil'): true_probdist,
            ('pos', 'bunnies'): true_probdist,
            ('pos', 'butteryfly'): true_probdist,
            ('pos', 'pony'): true_probdist,
            ('pos', 'love'): true_probdist,
            ('pos', 'smile'): true_probdist,
            ('pos', 'happy'): true_probdist,
            ('pos', 'amazing'): true_probdist,
            ('pos', 'yes'): true_probdist,
}
 
 
classifier = NaiveBayesClassifier(label_probdist, feature_probdist)
 
for sample in test_samples:
    print "%s | %s | %s" % (sample, classifier.classify(gen_bow(sample)), classifier.prob_classify(gen_bow(sample)))
 
classifier.show_most_informative_features()
