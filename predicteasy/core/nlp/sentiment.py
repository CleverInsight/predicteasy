from textblob import TextBlob




class SentimentClassifier:

	def __init__(self, text):
		self.data = text


	def __str__(self):
		return '{self.data} module'.format(self=self)

	
	def predict(self):

		_txt = TextBlob(self.data)
		
		if _txt.sentiment.polarity == 0:
			result = 'neutral'

		if _txt.sentiment.polarity > 0:
			result = 'positive'

		if _txt.sentiment.polarity < 0:
			result = 'negative'

		return dict(polarity=_txt.sentiment.polarity, 
			subjectivity=_txt.sentiment.subjectivity,
			sentiment=result) 