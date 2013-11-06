
import svm
from svmutil import *
import re, pickle, csv, os
import classifier_helper, html_helper

#start class
class SVMClassifier:
    """ SVM Classifier """
    #variables    
    #start __init__
    def __init__(self, tweetsDataFile, trainingDataFile, classifierDumpFile, iStopWords, iFeatureList, trainingRequired = 0):
    #def __init__(self, data, keyword, time, trainingDataFile, classifierDumpFile, trainingRequired = 0):
        #Instantiate classifier helper        
        self.helper = classifier_helper.ClassifierHelper('tweets_feature_list.txt')
        self.helperFullList = classifier_helper.ClassifierHelper('tweets_full_feature_list.txt')		
                     
        #read in and process our self supplied tweets        	
        self.tweetsDataFile = tweetsDataFile
        self.data = self.getDataFromFiles(self.tweetsDataFile)
        self.lenTweets = len(self.data)				
        self.origTweets = self.getUniqData(self.data)
        self.stopWords = iStopWords
        self.featureList = iFeatureList
        self.tweets = self.getProcessedTweets(self.origTweets)    			                   			     
        
				
        self.results = {}
        self.neut_count = [0] * self.lenTweets
        self.pos_count = [0] * self.lenTweets
        self.neg_count = [0] * self.lenTweets
        self.trainingDataFile = trainingDataFile

        #self.time = time
        #self.keyword = keyword
        self.html = html_helper.HTMLHelper()
        
        #call training model
        if(trainingRequired):
            self.classifier = self.getSVMTrainedClassifer(trainingDataFile, classifierDumpFile)
        else:
            fp = open(classifierDumpFile, 'r')
            if(fp):
                self.classifier = svm_load_model(classifierDumpFile)
            else:
                self.classifier = self.getSVMTrainedClassifer(trainingDataFile, classifierDumpFile)
    #end
    
    #start getUniqData
    def getUniqData(self, data):
        uniq_data = {}        
        for i in data:
            d = data[i]
            u = []
            for element in d:
                if element not in u:
                    u.append(element)
            #end inner loop
            uniq_data[i] = u            
        #end outer loop
        return uniq_data
    #end
    
    #start getProcessedTweets
    def getProcessedTweets(self, data):        
        tweets = {}        
        for i in data:
            d = data[i]
            tw = []
            for t in d:
                tw.append(self.helperFullList.process_tweet_stopwords(t, self.stopWords))
            tweets[i] = tw            
        #end loop
        return tweets
    #end
    
	
    def getDataFromFiles(self, tweetsDataFile):
        fp = open( tweetsDataFile, 'rb' )
        reader = csv.reader(fp, delimiter='*', quotechar='|')
        weekTweets = {}
        tweets = []       
		
        for row in reader:             
           #print "tweet = %s\n" % (row[0]) 			  
           tweets.append(row[0])
           #end loop
		#end loop
        for i in range(0,1):
           weekTweets[i] = tweets
        #end loop
        
        print "SVM data set size: %s" % (len(tweets)) 			
        return weekTweets
    #end
	
	
    #start getNBTrainedClassifier
    def getSVMTrainedClassifer(self, trainingDataFile, classifierDumpFile):        
        # read all tweets and labels
        tweetItems = self.getFilteredTrainingData(trainingDataFile)
        
        tweets = []
        for (words, sentiment) in tweetItems:
            words_filtered = [e.lower() for e in words.split() if(self.helper.is_ascii(e))]
            tweets.append((words_filtered, sentiment))

        results = self.helper.getSVMFeatureVectorAndLabels(tweets)
        self.feature_vectors = results['feature_vector']
        self.labels = results['labels']
        
        #SVM Trainer
        problem = svm_problem(self.labels, self.feature_vectors)
        #'-q' option suppress console output		
        param = svm_parameter('-q')
        param.kernel_type = LINEAR
        #param.show()
        classifier = svm_train(problem, param)
        svm_save_model(classifierDumpFile, classifier)
        return classifier
    #end
    
    #start getFilteredTrainingData
    def getFilteredTrainingData(self, trainingDataFile):
        fp = open( trainingDataFile, 'rb' )
        min_count = self.getMinCount(trainingDataFile)
        min_count = 40000
        neg_count, pos_count, neut_count = 0, 0, 0
        
        reader = csv.reader(fp, delimiter='*', quotechar='|')
        #reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
        tweetItems = []
        #print 'inside getFilteredTrainingData\n'
        count = 1       
        for row in reader:
            #processed_tweet = self.helper.process_tweet(row[4])
            processed_tweet = self.helper.process_tweet_stopwords(row[1], self.stopWords)			
            sentiment = row[0]
            
            if(sentiment == 'neutral'):                
                if(neut_count == int(min_count)):
                    continue
                neut_count += 1
            elif(sentiment == 'positive'):
                if(pos_count == min_count):
                    continue
                pos_count += 1
            elif(sentiment == 'negative'):
                if(neg_count == int(min_count)):
                    continue
                neg_count += 1
            
            tweet_item = processed_tweet, sentiment
            tweetItems.append(tweet_item)
            count +=1
        #end loop
        return tweetItems
    #end 

    #start getMinCount
    def getMinCount(self, trainingDataFile):
        fp = open( trainingDataFile, 'rb' )
        reader = csv.reader(fp, delimiter='*', quotechar='|')
        #reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
        neg_count, pos_count, neut_count = 0, 0, 0
        for row in reader:
            sentiment = row[0]
            if(sentiment == 'neutral'):
                neut_count += 1
            elif(sentiment == 'positive'):
                pos_count += 1
            elif(sentiment == 'negative'):
                neg_count += 1
        #end loop
        return min(neg_count, pos_count, neut_count)
    #end

    #start classify
    def classify(self):
        for i in self.tweets:
            tw = self.tweets[i]
            test_tweets = []
            res = {}
            for words in tw:
                #words_filtered = [e.lower() for e in words.split() if(self.helper.is_ascii(e))]
                words_filtered = [e.lower() for e in words.split() if(self.helperFullList.is_ascii(e))]				
                test_tweets.append(words_filtered)
            #test_feature_vector = self.helper.getSVMFeatureVector(test_tweets)            
            test_feature_vector = self.helperFullList.getSVMFeatureVector(test_tweets)            
            p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),\
                                                test_feature_vector, self.classifier)
            count = 0
            for t in tw:
                label = p_labels[count]
                if(label == 0):
                    label = 'positive'
                    self.pos_count[i] += 1
                elif(label == 1):
                    label = 'negative'
                    self.neg_count[i] += 1
                elif(label == 2):
                    label = 'neutral'
                    self.neut_count[i] += 1
                result = {'text': t, 'tweet': self.origTweets[i][count], 'label': label}
                res[count] = result
                count += 1
            #end inner loop
            self.results[i] = res
        #end outer loop
    #end
           
    #start writeOutput
    def writeOutput(self, filename, writeOption='w'):
        fp = open(filename, writeOption)
        for i in self.results:
            res = self.results[i]
            for j in res:
                item = res[j]
                text = item['text'].strip()
                label = item['label']
                writeStr = text+" | "+label+"\n"
                fp.write(writeStr)
            #end inner loop
        #end outer loop      
    #end writeOutput    

    #start accuracy
    def accuracy(self):
                
        tweets = self.getFilteredTrainingData(self.trainingDataFile)
        test_tweets = []
		
		#find our training and test size
        num_folds = 10
        subset_size = len(tweets)/num_folds
		
        for i in range(num_folds):
          testing_this_round = tweets[i*subset_size:][:subset_size]
          training_this_round = tweets[:i*subset_size] + tweets[(i+1)*subset_size:]		
		
          for (t, l) in testing_this_round:
              words_filtered = [e.lower() for e in t.split() if(self.helper.is_ascii(e))]
              test_tweets.append(words_filtered)
		  
          test_feature_vector = self.helper.getSVMFeatureVector(test_tweets)        
		
          p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),\
                                            test_feature_vector, self.classifier)
          count = 0
          total , correct , wrong = 0, 0, 0
          self.accuracy = 0.0
          for (t,l) in testing_this_round:
              label = p_labels[count]
			
              if(label == 0):
                  label = 'positive'
              elif(label == 1):
                  label = 'negative'
              elif(label == 2):
                  label = 'neutral'

              if(label == l):
                  correct+= 1
              else:
                  wrong+= 1
              total += 1
              count += 1
          #end loop

          test_tweets = []
          self.accuracy = (float(correct)/total)*100
          print 'fold= %d, total = %d, correct = %d, wrong = %d, accuracy = %.2f' % \
                                                (i, total, correct, wrong, self.accuracy)        														  
        #end loop
    #end

    #start getHTML
    def getHTML(self):
        return self.html.getResultHTML(self.keyword, self.results, self.time, self.pos_count, \
                                       self.neg_count, self.neut_count, 'svm')
    #end
#end class
