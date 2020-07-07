from pprint import pprint as print
from gensim.summarization import summarize


class TextSummarize:

	def __init__(self, text):
		self.data = text


	def summary(self, **kwargs):
		return summarize(self.data, **kwargs)