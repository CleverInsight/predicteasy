import io
import pandas as pd
from textblob import TextBlob

class SpellCheck:

	def __init__(self, text, multiple=False, column=""):
		self.data = text
		self.multiple = multiple
		self.column = column



	def spell_apply(self, data):
		"""
		spell_apply takes incorrect text and 
		correct the spell and returns it
		
		:param      data:  The data
		:type       data:  { type_description }
		
		:returns:   { description_of_the_return_value }
		:rtype:     { return_type_description }
		"""
		return str(TextBlob(data).correct())

	def correct(self):
		"""
		Correct method helps us to loop throught the 
		given dataframe and correct the grammer
		
		:returns:   { description_of_the_return_value }
		:rtype:     { return_type_description }
		"""
		if self.multiple:
			data = pd.read_csv(io.StringIO(self.data), lineterminator='\n')
			data.rename(columns=lambda x: x.strip(), inplace=True)
			data[self.column.strip()].apply(self.spell_apply).apply(pd.Series)
			return dict(data=data.to_json(orient='records'), summary=[])
		else:
			return self.spell_apply(self.data)