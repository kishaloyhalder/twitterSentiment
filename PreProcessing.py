import csv
import nltk
from nltk.corpus import stopwords
import re,os
from WordBreaker import WordBreaker

class PreProcessing:
	isNorm = True
	normDic = {} 
	emoticonList = {} 
	stopwordList = []  
	abrevDic = {}
	wordbreaker = WordBreaker()

	def loadEmoticons(self):
		dicPath = 'data/emoticons.csv'
		dic = {}
		with open(dicPath,'r') as csvfile:
			headers = csvfile.readline()
			fw = csv.reader(csvfile, delimiter='\t')
			for row in fw:
				items = row[2].split(' ')
				for item in items:
					dic[item] = row[3]
		return dic

		
	def loadNormDic(self):
		dicPath = 'data/norm_dic.txt'
		dic = {}

		if not os.path.exists(dicPath):
			print('load from old')
			
			dicPath1 = 'D:/data/emnlp/emnlp_dict.txt'
			dicPath2 = 'D:/data/lexnorm/lexnorm/data/corpus.tweet'
			
			
			#load dic #1
			with open(dicPath1,'r') as csvfile:
				fw = csv.reader(csvfile, delimiter='\t')
				for row in fw:
					if not row[0] == row[1]:
						dic[row[0]] = row[1]

			#load dic #2
			with open(dicPath2,'r') as csvfile:
				fw = csv.reader(csvfile, delimiter='\t')
				for row in fw:
					if len(row) == 3:
						if not row[0] == row[2]:
							dic[row[0]] = row[2]
			writeDic2File(dic)
		else:
			#print 'load from new'
			#load dic 
			with open(dicPath,'r') as csvfile:
				fw = csv.reader(csvfile, delimiter='\t')
				for row in fw:
					dic[row[0]] = row[1]
		return dic

	def loadAbrevDic(self):
		path = 'data/abbreviationDic.txt'
		dic = {}
		with open(path,'r') as fr:
			lines = fr.readlines()
			for line in lines:
				items = line.split('\t')
				dic[items[0]] = items[1]
		return dic

	def removeRT_Mention_hashsign_url(self,content):
		content.encode('utf-8')
		hashtag = []
		hashtagPattern = '(#\w+)'
		for m in re.finditer(hashtagPattern,content):
			normalizedHashtag = self.wordbreaker.breakIntoWords(m.group(0))
			if normalizedHashtag == '':
				hashtag.append(m.group(0)[1:])
			else:
				hashtag.append(normalizedHashtag)
				content = re.sub(m.group(0), '#'+normalizedHashtag, content)
		#remove RT|@...|#|url
		pattern = r'(RT)|(@\w+[:]*)|(http\S+)|(#)'
		content = re.sub(pattern, '', content)
		return [content,hashtag]

	def removeRT_Mention_hashtag_url(self,content):
		content.encode('utf-8')
		hashtag = []
		hashtagPattern = '(#\w+)'
		for m in re.finditer(hashtagPattern,content):
			normalizedHashtag = self.wordbreaker.breakIntoWords(m.group(0))
			if normalizedHashtag == '':
				hashtag.append(m.group(0)[1:])
			else:
				hashtag.append(normalizedHashtag)
		#remove RT|@...|#|url
		pattern = r'(RT)|(@\w+[:]*)|(#\w+)|(http\S+)'
		content = re.sub(pattern, '', content)
		return [content,hashtag]

	def removeStopword_spellCheck_abrevExpand(self,content): 
		# split word
		content = re.sub(r'(\")',' \" ',content)
		word_list = nltk.word_tokenize(content)
		# remove stopword                    
		filtered_words = []    
		for w in word_list:
			if w in self.abrevDic:
				w = self.abrevDic[w].split()
			else:
				#for the ease of unified format, because the former is list
				w = w.split()
			for e_w in w:
				if not e_w in self.stopwordList:
					if e_w in self.normDic:
						e_w = self.normDic[e_w]
					filtered_words.append(e_w)
		return filtered_words  
		  
	def check_remove_emoticons(self,content):
		#check and remove emoticons begin
				emo_pos = 0
				emo_neg = 0
				for emo in self.emoticonList:
					if content.find(emo) > -1:
						content = content.replace(emo,'')
						if self.emoticonList[emo] == 'Positive':
							emo_pos += 1
						else:
							emo_neg += 1
				#check and remove emoticons end
				return [content,emo_pos,emo_neg]

	# input:
	#    content: the initial tweet you want to normalize
	#    withHashTag: True if you want to keep the hashtag (without '#'), otherwise remove the hashtag
	# output:
	#    [content,emo_pos, emo_neg]
	#    content: normalized result
	#    emo_pos: # positive emoticons
	#    emo_neg: # negtive emoticons
	def normalizeTweet(self,content, withHashTag):
		# to lower case
		#content = content.lower()
		
		if withHashTag is True:
			#only remove # of the hashtag
			[content,hashtag] = self.removeRT_Mention_hashsign_url(content)
		else:     
			#remove the content of the hashtag
			[content,hashtag] = self.removeRT_Mention_hashtag_url(content)
			
		#must first remove the link because :/ is negative emoticons but in http://
		[content, emo_pos, emo_neg] = self.check_remove_emoticons(content)
		
		content = self.removeStopword_spellCheck_abrevExpand(content)
		return [content,emo_pos, emo_neg, hashtag]
	
	def __init__(self):
		#load all the files as needed
		#stopwordList = stopwords.words('english')
		self.normDic = self.loadNormDic()
		self.emoticonList = self.loadEmoticons()
		self.abrevDic = self.loadAbrevDic()



	#usage example
	#tweet = '143 4n RT @angelsmomaw: :) :-) #HCR is unwanted because it will bankrupt the USA and give below inferior Healthcare for all. #gop #tcot #tweetcongress http://bit.ly/9Klc8V #hcr'
	#[content,emo_pos, emo_neg] = normalizeTweet(tweet,True)
	#print([content,emo_pos, emo_neg])

	#[content,emo_pos, emo_neg] = normalizeTweet(tweet,False)
	#print([content,emo_pos, emo_neg])

