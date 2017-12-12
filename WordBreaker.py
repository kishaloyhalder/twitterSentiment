from os import listdir
from os import path

class WordBreaker:
	validWords = {}
	words = []
	
	def __init__(self):
		self.initialize()
	
	def initialize(self):
		scriptDir = path.dirname(path.abspath(__file__))
		dictionaryDir = scriptDir + '\\data\\dictionary\\'
		files = listdir(dictionaryDir)
		for file in files:
			#print 'reading' + file
			fileContent = open(dictionaryDir + file, 'r', encoding="utf8")
			for line in fileContent:
				self.validWords[line.rstrip('\n').lower()] = 1
		

	def isWord(self, s):
		return s in self.validWords

	def wordBreak(self, s):
		length = len(s)
		if(self.isWord(s)):
			self.words.append(s)
			return True
			
		else:
			for x in range(0, length):
				sstr = s[:x]
				isLHSWord = self.isWord(sstr)
				if(isLHSWord):
					isRHSWord = self.wordBreak(s[x:])
					if (isRHSWord):
						self.words.append(sstr)
						return True
		return False
		
	def printWords(self):
		self.words.reverse()
		segmentedWords = ''
		for i in range(0, len(self.words)):
			segmentedWords = segmentedWords + self.words[i] + ' '
		self.words = []
		return segmentedWords
	
	def cleanUp(self, s):
		s = s.lower();
		s = ''.join(e for e in s if e.isalpha())
		return s
		
	def breakIntoWords(self, s):
		s = self.cleanUp(s)
		if (self.wordBreak(s)):
			return self.printWords()
		else:
			return ''



#instance = WordBreaker()

#print instance.breakIntoWords('When I text someone and they')

