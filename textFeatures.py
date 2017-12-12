import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
import sys
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import _pickle as pickle


from PreProcessing import PreProcessing


class textFeatures:
	def getText(self,normalized):
		return ' '.join(normalized[0])
		
		
	def getHashTag(self,normalized):
		return ','.join(normalized[3])


	def getPrintableString(self,normalized):
		return '+++'.join([self.getText(normalized),str(normalized[1]),str(normalized[2]),self.getHashTag(normalized)])
		
	
	def getStackedPrediction(self,tweets):
		process = PreProcessing()
		## load model for content classifier
		with open('models\\tfIdfClassifierForContent.pkl', 'rb') as fid:
			contentClassifier = pickle.load(fid)
			
		## load model for hashtag classifier
		with open('models\\tfIdfClassifierForHashTag.pkl', 'rb') as fid:
			hashtagClassifier = pickle.load(fid)
		## Feature Extraction
		
		tweets['normalized'] = tweets.content.apply(process.normalizeTweet,args=(True,))
		tweets['normalizedText'] = tweets.normalized.apply(self.getText)
		tweets['hashtag'] = tweets.normalized.apply(self.getHashTag)
		
		test = np.array(tweets)
		
		indexTonormalizedText = tweets.columns.get_loc('normalizedText')
		indexTohashtag = tweets.columns.get_loc('hashtag')
		
		## Predicting for TfIdfContent

		preds = contentClassifier.predict(test[:,indexTonormalizedText])
		tweets['stackedPredictionFromContentTfIdf'] = preds
		
		## Predicting for TfIdfHashTag

		preds = hashtagClassifier.predict(test[:,indexTohashtag])
		tweets['stackedPredictionFromHashTagTfIdf'] = preds
		
		tweets['normalized'] = tweets.normalized.apply(self.getPrintableString)
		return tweets
		
	
	def trainTextFeatureClassifier(self):
		process = PreProcessing()

		tweets =  pd.read_csv('dataset/tweetDataset_cleaned.csv')

		#Setting training data and test data indices
		type = np.array(tweets.type)
		train_idx = np.where(type == 'train')[0]
		test_idx = np.where(type == 'test')[0]



		tweets['normalized'] = tweets.content.apply(process.normalizeTweet,args=(True,))
		tweets['normalizedText'] = tweets.normalized.apply(self.getText)
		tweets['hashtag'] = tweets.normalized.apply(self.getHashTag)


		train = np.array(tweets[tweets['type'] == 'train'])
		test = np.array(tweets)
		labels = np.array(tweets['sentiment'])[train_idx]

		pipeline = Pipeline([
			('tfidf', TfidfVectorizer()),
			('clf', MultinomialNB()),
		])

		parameters = {
			'tfidf__max_df': (0.7, 0.8, 1.0),
			'tfidf__min_df': (0.0, 0.1, 0.2),
			'tfidf__max_features': (None, 5000, 10000, 20000, 50000),
			'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
			'tfidf__norm': ('l1', 'l2'),
			'clf__alpha': (0.000003,0.0003, 0.003, 0.03),
			'clf__class_prior': ([0.4,0.4,0.2],[0.3,0.3,0.4],[0.33,0.33,0.33],None)
		}


		if __name__ == "__main__":
			grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0)

			print("Performing grid search...")
			print("pipeline:", [name for name, _ in pipeline.steps])
			print("parameters:")
			pprint(parameters)
			t0 = time()
			grid_search.fit(train[:,9], labels)


			print("done in %0.3fs" % (time() - t0))
			print()

			print("Best score for TfIdfContent: %0.3f" % grid_search.best_score_)
			print("Best parameters set:")
			best_parameters = grid_search.best_estimator_.get_params()
			for param_name in sorted(parameters.keys()):
				print("\t%s: %r" % (param_name, best_parameters[param_name]))
				
			# save the model
			with open('tfIdfClassifierForContent.pkl', 'wb') as fid:
				pickle.dump(grid_search, fid)  
				
			## Predicting for TfIdfContent

			preds = grid_search.predict(test[:,9])

			tweets['stackedPredictionFromContentTfIdf'] = preds
				
			
			print("Performing grid search...")
			print("pipeline:", [name for name, _ in pipeline.steps])
			print("parameters:")
			pprint(parameters)
			t0 = time()
			grid_search.fit(train[:,10], labels)


			print("done in %0.3fs" % (time() - t0))
			print()

			print("Best score for TfIdfHashTag: %0.3f" % grid_search.best_score_)
			print("Best parameters set:")
			best_parameters = grid_search.best_estimator_.get_params()
			for param_name in sorted(parameters.keys()):
				print("\t%s: %r" % (param_name, best_parameters[param_name]))
				
			# save the model
			with open('tfIdfClassifierForHashTag.pkl', 'wb') as fid:
				pickle.dump(grid_search, fid)  
				
			## Predicting for TfIdfHashTag

			preds = grid_search.predict(test[:,10])

			tweets['stackedPredictionFromHashTagTfIdf'] = preds
			
			tweets['normalized'] = tweets.normalized.apply(self.getPrintableString)

			tweets.to_csv('stackedPredictionFromTfIdfUsingGridSearch.csv', index = False)
			

			
#instance = textFeatures()

#instance.trainTextFeatureClassifier()