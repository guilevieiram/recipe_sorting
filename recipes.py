import pandas as pd
import numpy as np
import os, spacy
import sqlite3
from operator import add
from functools import reduce

from sklearn.model_selection import train_test_split
import torch
import transformers
from transformers import BertForSequenceClassification, BertTokenizer
'''
To install latest version of HuggingFace transformers:
pip install git+https://github.com/huggingface/transformers
You may need to:
pip install sentencepiece
'''

nlp = spacy.load("en_core_web_sm")
'''
to install spacy sm module run on comand line:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
or if you're using conda, run:
conda install -c conda-forge spacy-model-en_core_web_sm

to install larger spacy model (en_core_web_lg) run on terminal:
python -m spacy download en_core_web_lgs
'''
	
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
	
	# Main methods
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

		self.prep_data()

	def word_frequency_analysis (self):
		# Saving all lemmatized methods to a doc
		doc = []
		for description in self.evaluation_data["recipe_methods"]["description"]:
			doc += lemmatize(str(description))

		# Finding distribution on absolute and relative frequency
		word_distribution_series = pd.DataFrame(doc, columns = ["words"]).value_counts()
		self.word_distribution = word_distribution_series.to_frame(name = "frequency")
		total = self.word_distribution.shape[0]
		self.word_distribution = self.word_distribution.assign(relative_frequency = self.word_distribution['frequency']/total)

		# Classifying words
		self.word_distribution = self.word_distribution.assign(type = self.classify_word(self.word_distribution.index))

		# Exporting everything to csv
		self.word_distribution.to_csv(r'word_frequency_analysis/word_frequency.csv')
		self.word_distribution[self.word_distribution['type'] == 'ingredients'].to_csv(r'word_frequency_analysis/word_frequency_ingredients.csv')
		self.word_distribution[self.word_distribution['type'] == 'tools'].to_csv(r'word_frequency_analysis/word_frequency_tools.csv')
		self.word_distribution[self.word_distribution['type'] == 'techniques'].to_csv(r'word_frequency_analysis/word_frequency_techniques.csv')


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

	def classify_word(self, words):
		# transforming listing into sets for O(1) search
		ingredients_list = list(map(lemmatize, set(self.listings['ingredients']['name'])))
		ingredients_set = set(reduce(add ,ingredients_list))

		tools_list = list(map(lemmatize, set(self.listings['tools']['name'])))
		tools_set = set(reduce(add ,tools_list))

		techniques_list = list(map(lemmatize, set(self.listings['techniques']['name'])))
		techniques_set = set(reduce(add ,techniques_list))

		# classifying according to listing
		results = []
		for word in words:
			word = word[0]
			if word in ingredients_set: results.append('ingredients')
			elif word in tools_set: results.append('tools')
			elif word in techniques_set: results.append('techniques')
			else: results.append('others')
		return results

	def predict_techniques(self, description, model, tokenizer, treshold=0.5):
		# create techniques_list
		techniques_list = list(self.listings['techniques']['name'])

		# Tokenize inputs
		input = tokenizer(description, return_tensors='pt')
		# Initializes labels
		labels = torch.LongTensor([0])
		# Calculate outputs from given model
		output = model(**input, labels=labels)
		# Logist values from the output (without gradient)
		logits = output.logits.detach().squeeze()
		# Calculating probabilities from output
		probabilities = torch.sigmoid(logits)
		# Passing proabilities through treshold
		predicted_bools = [(prob>treshold).item() for prob in probabilities]
		# Assigning list with predicted techiques
		predicted_techniques = [technique for (technique, boolean) in zip(techniques_list, predicted_bools) if boolean]

		return predicted_techniques


	# Prepping data methods
	def prep_data_classapp(self):

		# Creating techniques df
		techniques_df = self.listings['techniques'][['name', 'badge_id', 'badge_name']]
		techniques_df.rename_axis('techniques_id', inplace=True)

		# Creating badges df
		badges_df = self.listings['badges'][self.listings['badges']['category']=='technique']
		badges_df = badges_df[['name']]
		badges_df.rename_axis('badges_id', inplace=True)

		# Creating methods df
		methods_df = self.evaluation_data['recipe_methods']
		methods_df.rename_axis('methods_id', inplace=True)

		# Creating 'method index' column
		methods_df['method_index'] = 0
		num_indexes = methods_df.shape[0]
		for index in range(1,num_indexes):
			if methods_df.at[index, 'recipe_id'] == methods_df.at[index - 1, 'recipe_id']:
				methods_df.at[index, 'method_index'] = methods_df.at[index - 1, 'method_index'] + 1
			else:
				methods_df.at[index, 'method_index'] = 1

		# Creating recipes df
		recipes_df = self.evaluation_data['recipes'][['title']]
		recipes_df.rename_axis('recipes_id', inplace=True)	


		# Shuffling recipes
		np.random.RandomState(seed=2021)
		recipes = [df for _, df in methods_df.groupby('recipe_id')]
		np.random.shuffle(recipes)
		df = pd.concat(recipes).reset_index()
		df.rename_axis('rand_method_id', inplace = True)
		rand_methods_df = df

		# Exporting pickles
		path = r'c:/users/guilh/code/classapp/data'
		techniques_df.to_pickle(path + '/techniques')
		badges_df.to_pickle(path + '/badges')
		rand_methods_df.to_pickle(path + '/methods')
		recipes_df.to_pickle(path+ '/recipes')

	def prep_data_bert(self):
		# Defining user names just as in the ClassApp
		user_list = ['GV', 'NA', 'RC', 'WZ', 'EM', 'FR', 'HC', 'JF', 'MG']
		db_names = ['data_' + user + '.db' for user in user_list]

		# Creating paths to their .db files in the classapp_output folder
		db_paths = [os.path.join('classapp_output', db_name) for db_name in db_names]

		# Creating a list of data frames from all users
		data_frames = []

		sql = '''
		SELECT * FROM class_methods
		'''
		for db_path in db_paths:
			if os.path.exists(db_path):
				conn = sqlite3.connect(db_path)
				data_frames.append(pd.read_sql_query(sql, conn))
				conn.commit()
				conn.close()

		# Concatenating that list in one data frame
		users_methods = pd.concat(data_frames)
		
		# Saving description by id to later reinsertion
		description_by_method_id = users_methods[["method_id", "description"]].drop_duplicates().set_index('method_id')

		# Summing classifications for each value (to get multi-label classification)
		vectorized_methods = users_methods.groupby('method_id').sum()

		# Joining the descriptions
		classified_methods = pd.concat([description_by_method_id, vectorized_methods], axis='columns').reset_index(drop=True)

		# Listing the methods
		techniques = list(classified_methods.columns[1:])

		# Fixing anomalies in the table
		for technique in techniques:
			classified_methods.loc[classified_methods[technique] > 1, technique] = 1
			classified_methods.loc[classified_methods[technique] < 0, technique] = 0

		# Separating data in train and testing
		df_train, df_test = train_test_split(classified_methods, test_size=0.2, random_state=42)

		# Pickling and csving the dfs
		classified_methods.to_pickle(os.path.join('hand_classified_methods', 'classified_methods'))
		
		classified_methods.to_csv(os.path.join('hand_classified_methods', 'classified_methods.csv'), index=False)
		df_train.to_csv(os.path.join('hand_classified_methods', 'methods_train.csv'), index=False)
		df_test.to_csv(os.path.join('hand_classified_methods', 'methods_test.csv'), index=False)

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

	def prep_data(self):

		# Fixing casting types on evaluation methods
		self.evaluation_data['recipe_methods'].dropna(axis='index', inplace=True)
		self.evaluation_data['recipe_methods'].index = self.evaluation_data['recipe_methods'].index.astype('int')
		self.evaluation_data['recipe_methods'] = self.evaluation_data['recipe_methods'].astype({'recipe_id': 'int'})

		# Assigning badge name to techniques
		self.listings['techniques'] = self.listings['techniques'].assign(badge_name = self.find_badge_name(self.listings['techniques']['badge_id']))

	# Sorting methods

	# HAVENT FOUND ANYTHING COOL TO USE HERE YET AND IM NOT PRONED TO DO IT MYSELF	
	def sort_ingredients(self):
		pass
	
	
	# dont know how to write a good method for this as well. similarity? NN? who knows
	def sort_tools(self):
		pass
	

	# probably will be made using transformers model
	def sort_techniques(self):

		# Set up model in the CPU (this could be changed to gpu for better performance)
		path = r'model/BERT_model'
		model = torch.load(path, map_location=torch.device('cpu'))
		model.eval()

		# Set up tokenizer
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

		# Set up mapping function
		mapping = lambda df: self.predict_techniques(df, model, tokenizer)

		# Predicting and mapping on output dataframe
		predictions = self.evaluation_data['recipe_methods']['description'][0:10].map(mapping)

		# Format for self.output dataframe
		self.output_data['recipe_techniques'] = pd.DataFrame(columns=['recipe_id', 'method_id', 'technique'])

		# Insert techniques
		for method_id in range(predictions.shape[0]):
			for technique in predictions[method_id]:
				self.output_data['recipe_techniques'] = self.output_data['recipe_techniques'].append({
					'method_id': method_id,
					'technique': technique
				}, ignore_index=True)

		# Assigning recipe id 
		assign_recipe_id = lambda x: self.evaluation_data['recipe_methods'].at[x, 'recipe_id']
		self.output_data['recipe_techniques']['recipe_id'] = self.output_data['recipe_techniques']['method_id'].map(assign_recipe_id)

		# Assigning method description
		assign_method_description = lambda x: self.evaluation_data['recipe_methods'].at[x, 'description']
		self.output_data['recipe_techniques']['description'] = self.output_data['recipe_techniques']['method_id'].map(assign_method_description)


# MISCELLANEOUS FUNCTIONS
def lemmatize (evaluation_string):
    doc = nlp(evaluation_string)
    lemmatized = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return lemmatized


if __name__ == "__main__":

	cuukin = Recipes()

	cuukin.import_data(training_folder = "training", listings_folder = "listings", evaluation_folder='evaluation')

	# cuukin.word_frequency_analysis()

	# cuukin.prep_data_classapp()

	# cuukin.prep_data_bert()

	cuukin.sort_techniques()