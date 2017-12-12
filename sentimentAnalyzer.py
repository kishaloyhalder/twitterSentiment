import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
import sys
import _pickle as pickle
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import coo_matrix
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from pprint import pprint
from time import time

from textFeatures import textFeatures
from ArticleSentimentAnalyzer import ArticleSentimentAnalyzer

class sentimentAnalyzer:

	strongPositiveWords = dict()
	strongNegativeWords = dict()
	neutralWords = dict()
	weakPositiveWords = dict()
	weakNegativeWords = dict()
	polarityList = ''
	tweets_numeric_columns = []
	tweets_categorial_columns = []
	
	def __init__(self):
		self.polarityList = pd.read_csv('data/polarity.tff',delim_whitespace=True)
		self.readPolarity()
		self.tweets_numeric_columns = [
									'nounCount',
									'adjCount',
									'advCount',
									'verbCount',
									'#hashtags',
									'#capitalizedWords',
									'#strongPositiveWords',
									'#strongNegativeWords',
									'#weakPositiveWords',
									'#weakNegativeWords',
									'#positiveEmoticons',
									'#negativeEmoticons',
									'fractionOfNegativeSentencsInURL',
									'fractionOfPositiveSentencsInURL',
									'fractionOfNeutralSentencsInURL'
								]
		self.tweets_categorial_columns = [
										'stackedPredictionFromContentTfIdf',
										'stackedPredictionFromHashTagTfIdf',
										'isRetweet',
										'hasUserMention',
									]
		return

	def readPolarity(self):
		strongPositivePolarityList= self.polarityList[(self.polarityList['priorpolarity'] == 'priorpolarity=positive') & (self.polarityList['type'] == 'type=strongsubj')]
		self.strongPositiveWords.update(dict(zip(strongPositivePolarityList['word'].apply(lambda x: x.split('=')[1]),strongPositivePolarityList['pos'].apply(lambda x: x.split('=')[1]))))
		
		weakPositivePolarityList= self.polarityList[(self.polarityList['priorpolarity'] == 'priorpolarity=positive') & (self.polarityList['type'] == 'type=weaksubj')]
		self.weakPositiveWords.update(dict(zip(weakPositivePolarityList['word'].apply(lambda x: x.split('=')[1]),weakPositivePolarityList['pos'].apply(lambda x: x.split('=')[1]))))
		
		strongNegativePolarityList= self.polarityList[(self.polarityList['priorpolarity'] == 'priorpolarity=negative') & (self.polarityList['type'] == 'type=strongsubj')]
		self.strongNegativeWords.update(dict(zip(strongNegativePolarityList['word'].apply(lambda x: x.split('=')[1]),strongNegativePolarityList['pos'].apply(lambda x: x.split('=')[1]))))
		
		weakNegativePolarityList= self.polarityList[(self.polarityList['priorpolarity'] == 'priorpolarity=negative') & (self.polarityList['type'] == 'type=weaksubj')]
		self.weakNegativeWords.update(dict(zip(weakNegativePolarityList['word'].apply(lambda x: x.split('=')[1]),weakNegativePolarityList['pos'].apply(lambda x: x.split('=')[1]))))
		
		neutralPolarityList= self.polarityList[self.polarityList['priorpolarity'] == 'priorpolarity=neutral']
		self.neutralWords.update(dict(zip(neutralPolarityList['word'].apply(lambda x: x.split('=')[1]),neutralPolarityList['pos'].apply(lambda x: x.split('=')[1]))))
		

	def isNegated(self,content,word):
		window = 3
		negationWords = ['no','not','neither','nor','can\'t','doesn\'t','wouldn\'t','none','never']
		wordIndex = content.index(word)
		i = 1
		while i<window:
			if (wordIndex-i > 0) and (content[wordIndex-i] in negationWords):
				return True
			elif (wordIndex+i < len(content)) and (content[wordIndex+i] in negationWords):
				return True
			i = i+1
		return False
		
		
	def isTagEqual(self,tag1,tag2):
		tagDict = dict()
		
		tagDict['NN'] = 'noun'
		tagDict['JJ'] = 'adj'
		tagDict['RB'] = 'adverb'
		tagDict['VB'] = 'verb'
		
		if (tag1 == 'anypos') or (tag2 == 'anypos'):
			return True
		elif ((tag1[:2] in tagDict) and (tagDict[tag1[:2]] == tag2)) or ((tag2[:2] in tagDict) and (tagDict[tag2[:2]] == tag1)):
			return True
		else:
			return False
		

	def getPolarityCount(self,text,polairty):
		content = []
		stemmedWords = dict()
		wordCounter = 0
		
		for word in text:
			loweredWord = word[0].lower()
			text[wordCounter] = (loweredWord, word[1])
			wordCounter = wordCounter + 1
			stemmedWords[loweredWord] = self.stemWord(loweredWord)
			content.append(loweredWord)
			
		if polairty=='strong_positive':
			polarWords = dict(self.strongPositiveWords)
			oppositePolarWords = dict(self.strongNegativeWords)
		elif polairty=='strong_negative':
			polarWords = dict(self.strongNegativeWords)
			oppositePolarWords = dict(self.strongPositiveWords)
		elif polairty=='weak_positive':
			polarWords = dict(self.weakPositiveWords)
			oppositePolarWords = dict(self.weakNegativeWords)
		elif polairty=='weak_negative':
			polarWords = dict(self.weakNegativeWords)
			oppositePolarWords = dict(self.weakPositiveWords)
		elif polairty=='neutral':
			polarWords = dict(self.neutralWords)
			
		polarWordsInSentence = set()
		oppositePolarWordsInSentence = set()
		
		for word in text:
			if ((word[0] in polarWords) and (self.isTagEqual(polarWords[word[0]], word[1]))) or ((stemmedWords[word[0]] in polarWords) and (self.isTagEqual(polarWords[stemmedWords[word[0]]], word[1]))):
				polarWordsInSentence.add(word[0])
				
		
		for word in text:
			if ((word[0] in oppositePolarWords) and (self.isTagEqual(oppositePolarWords[word[0]], word[1]))) or ((stemmedWords[word[0]] in oppositePolarWords) and (self.isTagEqual(oppositePolarWords[stemmedWords[word[0]]], word[1]))):
				oppositePolarWordsInSentence.add(word[0])

		polarityCount = 0
		for word in polarWordsInSentence:
			if self.isNegated(content,word) == False:
				polarityCount = polarityCount + 1
		
		for word in oppositePolarWordsInSentence:
			if self.isNegated(content,word) == True:
				polarityCount = polarityCount + 1
				
		return polarityCount

	def tagTweet(self,text):
		text = nltk.word_tokenize(text)
		taggedText = nltk.pos_tag(text)
		return taggedText
		
	def stemWord(self,word):
		stemmer = nltk.stem.porter.PorterStemmer()
		return stemmer.stem(word)
		
	def getPOSCount(self,taggedText,pos):
		counts = Counter(tag for word,tag in taggedText)
		if pos == 'noun':
			return counts['NN'] + counts ['NNS'] + counts['NNP'] + counts['NNPS']
		elif pos == 'verb':
			return counts['VB'] + counts ['VBD'] + counts['VBG'] + counts['VBN'] + counts['VBP'] + counts['VBZ']
		elif pos == 'adjective':
			return counts['JJ'] + counts ['JJR'] + counts['JJS']
		elif pos == 'adverb':
			return counts['RB'] + counts ['RBR'] + counts['RBS']
		else:
			return counts[pos]

	def getHashTagCount(self,text):
		tags = str(text).split(',')
		return len(tags)

			
	def getEmoticonCount(self,normalized,sent):
		if sent == 'pos':
			return int(normalized.split('+++')[1])
		else:
			return int(normalized.split('+++')[2])
			

	def getCapitalizedWordCount(self,text):
		words = text.split(' ')
		return len([word for word in words if word.isupper()])


	def isRetweet(self,text):
		if re.search(r'RT ',text) == None:
			return False
		else:
			return True
			
	def hasUserMention(self,text):
		if re.search(r'@\S+',text) == None:
			return False
		else:
			return True
			
			
	def getURL(self,text):
		pattern = r'(http\S+)'
		url = re.search(pattern,text)
		if url is None:
			return ''
		return url.group(0)
		

		
		
		
		
	def extractFeatures(self,tweets):
	
		articleSentAnalyzer = ArticleSentimentAnalyzer()

		tweets['#positiveEmoticons'] = tweets.normalized.apply(self.getEmoticonCount,args=('pos',))
		tweets['#negativeEmoticons'] = tweets.normalized.apply(self.getEmoticonCount, args=('neg',))

		tweets['#hashtags'] = tweets.hashtag.apply(self.getHashTagCount)
		tweets['#capitalizedWords'] = tweets.normalizedText.apply(self.getCapitalizedWordCount)
		tweets['isRetweet'] = tweets.content.apply(self.isRetweet)
		tweets['hasUserMention'] = tweets.content.apply(self.hasUserMention)

		## getting url features

		tweets['url'] = tweets.content.apply(self.getURL)
		tweets['urlFeatures'] = tweets.url.apply(articleSentAnalyzer.getSentimentForUrl)


		tweets['fractionOfNegativeSentencsInURL'] = tweets.urlFeatures.apply(lambda x: x[0])
		tweets['fractionOfPositiveSentencsInURL'] = tweets.urlFeatures.apply(lambda x: x[1])
		tweets['fractionOfNeutralSentencsInURL'] = tweets.urlFeatures.apply(lambda x: x[2])


		tweets['taggedText'] = tweets.normalizedText.apply(self.tagTweet)
		tweets['nounCount'] = tweets.taggedText.apply(self.getPOSCount,args=('noun',))
		tweets['adjCount'] = tweets.taggedText.apply(self.getPOSCount,args=('adjective',))
		tweets['advCount'] = tweets.taggedText.apply(self.getPOSCount,args=('adverb',))
		tweets['verbCount'] = tweets.taggedText.apply(self.getPOSCount,args=('verb',))

		tweets['#strongPositiveWords'] = tweets.taggedText.apply(self.getPolarityCount,args=('strong_positive',))
		tweets['#strongNegativeWords'] = tweets.taggedText.apply(self.getPolarityCount,args=('strong_negative',))


		tweets['#weakPositiveWords'] = tweets.taggedText.apply(self.getPolarityCount,args=('weak_positive',))
		tweets['#weakNegativeWords'] = tweets.taggedText.apply(self.getPolarityCount,args=('weak_negative',))

		return tweets

		
	def encodeFeatures(self,tweets):
	
		indicesToCategoricalFeatures = []
		tweets_numerical_values = np.array(tweets[self.tweets_numeric_columns])

		tweets_categorial_values = np.array(tweets[self.tweets_categorial_columns],dtype=str)


		with open('models\\label_encoder_column_'+str(0)+'.pkl', 'rb') as fid:
			label_encoder = pickle.load(fid)
		tweets_data = label_encoder.transform(tweets_categorial_values[:,0])
		indicesToCategoricalFeatures.append(0)


		for i in range(1, tweets_categorial_values.shape[1]):
			with open('models\\label_encoder_column_'+str(i)+'.pkl', 'rb') as fid:
				label_encoder = pickle.load(fid)
			tweets_data = np.column_stack((tweets_data, label_encoder.transform(tweets_categorial_values[:,i])))
			indicesToCategoricalFeatures.append(i)
			
		for i in range(0, tweets_numerical_values.shape[1]):
			tweets_data = np.column_stack((tweets_data, tweets_numerical_values[:,i]))

		#One hot encoding!
		with open('models\\one_hot_encoder.pkl', 'rb') as fid:
			enc = pickle.load(fid)
		tweets_data = enc.transform(tweets_data)

		tweets_data = tweets_data.astype(float)
		tweets_data = coo_matrix(tweets_data).tocsr()

		return tweets_data

	
	def trainClassifier(self,tweets):
	
		indicesToCategoricalFeatures = []
		tweets_numerical_values = np.array(tweets[self.tweets_numeric_columns])

		tweets_categorial_values = np.array(tweets[self.tweets_categorial_columns],dtype=str)


		label_encoder = LabelEncoder()
		label_encoder.fit(tweets_categorial_values[:,0])
		tweets_data = label_encoder.transform(tweets_categorial_values[:,0])
		indicesToCategoricalFeatures.append(0)
		with open('models\\label_encoder_column_'+str(0)+'.pkl', 'wb') as fid:
				pickle.dump(label_encoder, fid)  

		for i in range(1, tweets_categorial_values.shape[1]):
			label_encoder = LabelEncoder()
			label_encoder.fit(tweets_categorial_values[:,i])
			tweets_data = np.column_stack((tweets_data, label_encoder.transform(tweets_categorial_values[:,i])))
			indicesToCategoricalFeatures.append(i)
			with open('models\\label_encoder_column_'+str(i)+'.pkl', 'wb') as fid:
				pickle.dump(label_encoder, fid) 
				
				
		for i in range(0, tweets_numerical_values.shape[1]):
			tweets_data = np.column_stack((tweets_data, tweets_numerical_values[:,i]))

		#One hot encoding!
		enc = OneHotEncoder(categorical_features = indicesToCategoricalFeatures)
		enc.fit(tweets_data)
		tweets_data = enc.transform(tweets_data)
		with open('models\\one_hot_encoder.pkl', 'wb') as fid:
			pickle.dump(enc, fid)  
	
		
		tweets_data = tweets_data.astype(float)
		tweets_data = coo_matrix(tweets_data).tocsr()

		labels = np.array(tweets['sentiment'])


		parameters = {
			'alpha': (0.000003,0.0003, 0.003, 0.03),
			'class_prior': ([0.4,0.4,0.2],[0.3,0.3,0.4],[0.33,0.33,0.33],None)
		}

		if __name__ == "__main__":
			clf = MultinomialNB()
			grid_search = GridSearchCV(clf, parameters, n_jobs=-1, verbose=0)

			print("Performing grid search...")
			print("parameters:")
			pprint(parameters)
			t0 = time()
			grid_search.fit(tweets_data,labels)


			print("done in %0.3fs" % (time() - t0))
			print()

			print("Best score for SentimentAnalysis: %0.3f" % grid_search.best_score_)
			print("Best parameters set:")
			best_parameters = grid_search.best_estimator_.get_params()
			for param_name in sorted(parameters.keys()):
				print("\t%s: %r" % (param_name, best_parameters[param_name]))
				
				
			# save the model
			
			with open('models\\sentAnalsisClassifier.pkl', 'wb') as fid:
				pickle.dump(grid_search, fid)  
				
				
train = False
if train == True:
				
	trainTweets = pd.read_csv('dataset/train_cleaned.csv')
	instance = sentimentAnalyzer()
	tf = textFeatures()


	trainTweets = tf.getStackedPrediction(trainTweets)
	trainTweets = instance.extractFeatures(trainTweets)
	instance.trainClassifier(trainTweets)
				