import hashlib
import os
import json
import urllib.parse
import urllib.request
#import urllib3
from xml.dom import minidom

class ArticleSentimentAnalyzer:
	urls = {}
	articleTextsFile = ''
	delimiter = ' '
	lockFileName = 'urlSentiment.lc'
	fileLocation = 'articleTexts.txt'
	
	def __init__(self):
		if (self.lockFileExists()):
			self.loadFromLockFile()
		if os.path.isfile(self.fileLocation):
			articleFileContent = open(self.fileLocation,'r')
			for line in articleFileContent:
				fields = line.split('\t')
				url = fields[0]
				content = self.delimiter.join(fields[1:])
				#print 'processing url:' + url
				if (url in self.urls):
					sentiment = self.urls[url]
				else:
					sentiment = self.analyzeSentiment(content)
					self.urls[url] = sentiment
					self.updateLockFile(url, sentiment)
			articleFileContent.close()
			
	def updateLockFile(self, url, sentimentScores):
		lockFile = open(self.lockFileName, 'a')
		lockFile.write(url + '\t')
		#print sentimentScores
		for sentimentScore in sentimentScores:
			lockFile.write(str(sentimentScore))
			lockFile.write(',')
		lockFile.write('\n')
		lockFile.close()
			
	def lockFileExists(self):
		return os.path.isfile(self.lockFileName)
	
	def loadFromLockFile(self):
		lc = open(self.lockFileName,'r')
		for line in lc:
			fields = line.rstrip('\n').split('\t')
			url = fields[0]
			sentimentScores = fields[1]
			if sentimentScores == '':
				sentimentScores = '0.0,0.0,0.0,'
			listOfSentimentScores = self.getListFromStr(sentimentScores)
			self.urls[url] = listOfSentimentScores
		lc.close()
		print('done loading')
	
	def getListFromStr(self, content):
		fields = content.split(',')
		resultSet = []
		for field in fields:
			if (field != '' and field != ' '):
				resultSet.append(float(field))
		return resultSet
	
	def getSentimentForUrl(self, url):
		if url == '':
			return [0.0,0.0,0.0]
		elif(url in self.urls):
			return self.urls[url]
		else:
			sentimentScores = self.analyzeSentiment(self.getContentForUrl(url))
			self.urls[url] = sentimentScores
			self.updateLockFile(url, sentimentScores)
			return sentimentScores
	
	
	
	def getContentForUrl(self, url):
		baseUrl = 'http://access.alchemyapi.com/calls/url/URLGetText?apikey=392267b7953c4e61397cd92992fa64a47dbd3513&outputMode=json&url=';
		suffix = urllib.parse.quote(url.encode('utf-8'))
		queryUrl = baseUrl + suffix;
		response = urllib.request.urlopen(queryUrl).read().decode("utf-8")
		jsBody = json.loads(response)
		return jsBody['text']
		
	def shortenContent(self, content):
		#sentences = content.split('.')
		shortenedContent = ''
		#counter = 0
		if -1> 0:
			for sentence in sentences:
				shortenedContent = shortenedContent + ' ' + sentence
				counter = counter +1
				if counter > 5:
					break
		shortenedContent =  content[:5000]
		return shortenedContent
		
	def analyzeSentiment(self, content):
		resultSet = []
			
		if (content == ''):
			resultSet.append(-1.0)
			resultSet.append(-1.0)
			resultSet.append(-1.0)
			return resultSet
			
		content = self.shortenContent(content)
		numberOfSentencesToConsider = 5
		numberOfNeutralSentences = 0
		numberOfPositiveSentences = 0
		numberOfNegativeSentences = 0
		
		file = open('temp.txt','w', encoding='utf-8')
		
		if isinstance(content, bytes):
			content = content.decode('utf8')
		#print 'content is:' + content
		file.write(content)
		file.close()
		os.system('java -cp "stanford-corenlp-full-2015-01-29\*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,parse,sentiment -file temp.txt')
		xmldoc = minidom.parse('temp.txt.xml')
		sentenceList = xmldoc.getElementsByTagName('sentence')
		numberOfSentences = len(sentenceList)
		
		numberOfSentencesConsidered = 0
		
		for sentence in sentenceList:
			if sentence.attributes['sentiment'].value == 'Neutral':
				numberOfNeutralSentences = numberOfNeutralSentences + 1
			if (sentence.attributes['sentiment'].value == 'Positive' or sentence.attributes['sentiment'].value == 'Verypositive'):
				numberOfPositiveSentences = numberOfPositiveSentences + 1
			if (sentence.attributes['sentiment'].value == 'Negative' or sentence.attributes['sentiment'].value == 'Verynegative'):
				numberOfNegativeSentences = numberOfNegativeSentences + 1
				
			numberOfSentencesConsidered = numberOfSentencesConsidered + 1
			if (numberOfSentencesConsidered >= numberOfSentencesToConsider):
				break
		
		#print numberOfNegativeSentences
		#print numberOfNeutralSentences
		#print numberOfPositiveSentences
		#print numberOfSentences
		
		
		if numberOfSentencesConsidered != 0:
			resultSet.append(float(numberOfNegativeSentences)/numberOfSentencesConsidered)
			resultSet.append(float(numberOfPositiveSentences)/numberOfSentencesConsidered)
			resultSet.append(float(numberOfNeutralSentences)/numberOfSentencesConsidered)
			
		if -1>0:
			if numberOfSentences == 0:
				return -2	#invalid marker
			if (numberOfNegativeSentences> numberOfPositiveSentences):
				return -1
			if (numberOfPositiveSentences>numberOfNegativeSentences):
				return 1
			else:
				return 0
		return resultSet
		
	def getHashValue(self, descriptor):
		hashedValue = hashlib.sha224(descriptor).hexdigest()
		return hashedValue

#instance = ArticleSentimentAnalyzer()

#print 'done with the file *******'

#file = open('ursl.txt','r')
#for line in file:
#	print(instance.getSentimentForUrl(line.split('\t')[0]))
	
