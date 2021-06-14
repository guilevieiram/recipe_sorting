import pandas as pd, spacy
import numpy as np
import os, json, spacy
from collections import Counter

'''
to install spacy sm module run on comand line:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
or if you're using conda, run:
conda install -c conda-forge spacy-model-en_core_web_sm

to install larger spacy model (en_core_web_lg) run on terminal:
python -m spacy download en_core_web_lgs
'''

nlp = spacy.load("en_core_web_sm")
	
# MISCELLANEOUS FUNCTIONS
def lemmatize (evaluation_string):
    doc = nlp(evaluation_string)
    lemmatized = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return lemmatized

# MAIN CLASS
class Recipes():

	def __init__ (self):
		
		self.training_data = {
			"recipes": None,
			"recipe_methods": None,
			"recipe_ingredients": None,
			"recipe_tools": None,
			"recipe_techniques": None,

			"recipes_techniques_vector": None		
			}

		self.validation_data = {
			"recipes": None,
			"recipe_methods": None,
			"recipe_ingredients": None,
			"recipe_tools": None,
			"recipe_techniques": None			
			}

		self.evaluation_data = {
			"recipes": None,
			"recipe_methods": None,
			"recipe_ingredients": None		
			}
		
		self.output_data = {
			"recipe_ingredients": None,
			"recipe_tools": None,
			"recipe_techniques": None		
			}
		
		self.listings = {
			"ingredients": None,
			"tools": None,
			"techniques": None,
			"badges": None
			}
		
		self.word_distribution = None
	
	def import_data(self, listings_folder = None, training_folder = None, evaluation_folder = None):

		if listings_folder != None:
			for file_name in os.listdir(listings_folder):
				path = listings_folder + "//" + file_name
				table_name = os.path.splitext(file_name)[0]
				if file_name.endswith(".csv"):
					self.listings[table_name] = pd.read_csv(path)
				if file_name.endswith(".json"):
					self.listings[table_name] = pd.read_json(path_or_buf = path)
				self.listings[table_name].set_index("id", inplace = True)

		if training_folder != None:
			for file_name in os.listdir(training_folder):
				path = training_folder + "//" + file_name
				table_name = os.path.splitext(file_name)[0]
				if file_name.endswith(".csv"):
					self.training_data[table_name] = pd.read_csv(path)
				if file_name.endswith(".json"):
					self.training_data[table_name] = pd.read_json(path_or_buf = path)
				self.training_data[table_name].set_index("id", inplace = True)

		if evaluation_folder != None:
			for file_name in os.listdir(evaluation_folder):
				path = evaluation_folder + "//" + file_name
				table_name = os.path.splitext(file_name)[0]
				if file_name.endswith(".csv"):
					self.evaluation_data[table_name] = pd.read_csv(path)
				if file_name.endswith(".json"):
					self.evaluation_data[table_name] = pd.read_json(path_or_buf = path)
				self.evaluation_data[table_name].set_index("id", inplace = True)
	

	def word_frequency_analysis (self):
		
		doc = []
		
		for description in self.evaluation_data["recipe_methods"]["description"]:
			doc += lemmatize(description)

		word_distribution_series = pd.DataFrame(doc, columns = ["words"]).value_counts()
		self.word_distribution = word_distribution_series.to_frame(name = "frequency")
		total = self.word_distribution.shape[0]
		self.word_distribution = self.word_distribution.assign(relative_frequency = self.word_distribution['frequency']/total)

	
	def vectorize_techniques(self, groupcathegory='recipe_id'):
		self.training_data['recipe_techniques'] = self.training_data['recipe_techniques'].assign(
			technique_name = self.find_technique_name(self.training_data['recipe_techniques']['technique_id']))

		cathegories = self.listings['techniques']['name']
		dummies = pd.get_dummies(self.training_data['recipe_techniques']['technique_name'])
		missing_columns = list(set(cathegories)- set(dummies))

		number_rows = len(self.training_data['recipe_techniques'].index)
		missing_data_frame = pd.DataFrame(0, index=np.arange(1,number_rows+1), columns=missing_columns)

		merged = pd.concat([self.training_data['recipe_techniques'], dummies, missing_data_frame], axis='columns')
		merged.drop(columns=['created_at', 'updated_at', 'technique_name', 'technique_id'], inplace=True)

		grouped = merged.groupby([groupcathegory])
		grouped = grouped.sum()
		self.training_data["recipe_techniques_vector"] = grouped


	# assigning functions (should be called inside pandas assign method)
	def find_technique_name(self, id_list):
		result = []
		for id in id_list:
			result.append(self.listings['techniques']['name'][id])
		return result

	def find_badge_name(self, id_list):
		result = []
		for id in id_list:
			result.append(self.listings['badges']['name'][id])
		return result

	def count_number_methods(self, id_list):
		result = []
		num_methods_per_recipe = pd.DataFrame(self.evaluation_data['recipe_methods']['recipe_id'].value_counts())
		num_methods_per_recipe.rename(columns={'recipe_id': 'num_methods'}, inplace=True)

		for id in id_list:
			recipe_id = self.evaluation_data['recipe_methods'].at[id, 'recipe_id']
			result.append(num_methods_per_recipe.at[recipe_id, 'num_methods'])

		return result


	# HAVENT FOUND ANYTHING COOL TO USE HERE YET AND IM NOT PRONED TO DO IT MYSELF	
	def sort_ingredients(self):
		pass
	
	
	# dont know how to write a good method for this as well. similarity? NN? who knows
	def sort_tools(self):
		pass
	

	# probably will be made using transformers model
	def sort_techniques(self):
		pass

	

if __name__ == "__main__":

	cuukin = Recipes()
	cuukin.import_data(training_folder = "training", listings_folder = "listings", evaluation_folder='evaluation')
	cuukin.word_frequency_analysis(10)

		
	'''
	IDEAS:
	- for converting techniques names to binary values in several columns use onehot encoding (sklearn)
	- sklearn divides data in train and test
	
	'''