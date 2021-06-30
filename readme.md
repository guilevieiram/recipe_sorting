# Cuukin Recipe Sorting (using BERT models) 

This module has as a final goal to me able to classify recipe methods (e.g. "crack the eggs into a bowl and then fry them in a pan") into its culinary techniques (e.g. frying).
The main tool to accomplish this is BERT, a transformer model for Natural Language processing.
Many side tools were created to collect and prepare data for this task.

Here it is a brief summary of what the folders and the tools created are.

## recipes.py
This is our main file, containing the class Recipes. This class has all the working data as atributes and several methods for classification, preparation, analysis
Basically every other notebook or such derives from this class.

## Notebooks
All of the word done here was done on Jupyter notebooks and then implemented as methods in the recipes.py file.
These notebooks are of easier visualization if you don't understand whats happening on the actual code, and they are all in this notebooks folder

## DATA

### BBC_DB
Our analysis is done on top of this data base. It is a DB of around 6_000 recipes scraped from the BBC Foods website. The scraper organized recipes in three different tables:
- recipes.csv: Containing general information about each recipe (name, description, author, score, ...) + ID
- methods.csv: Contain the methods used in each recipe, with recipe ID as identification for that
- ingredients.csv: Contain the ingredients for each recipe. Those were parsed into name, quantity, unit and comments using a simple parser. This table is almost never used.

### evaluation
This is the evaluation data for the bert model. Basically the BBC data base with the right format and column names.
Tables:
- recipes
- recipe_methods
- recipe_ingredients

### listings
This are Json files with the listings of culinary ingredients, techniques and tools (all of which are considered "skills"). All were imported from the Cuukin App.
It also contains a Badges file. Badges work as cathegories for skills, such as fruits, pans or chopping techniques.

### training
This folder contains the hand classified data imported from the cuukin app.
This is a very small data set of only 12 recipes, a few original and others from BBC Foods.
Tables:
- recipes: general info on recipes
- recipe_methods: methods title and description for each recipe
- recipe_ingredients: ingredients (name, quantity, unit, comment) for each recipe
- recipe_tools: cooking tools for every recipe
- recipe_techniques: culinary techniques associated with each recipe

## Word Frequency Analysis
In order to start getting insight on the structure of the recipes I've setted up a simple frequency model.
This model analyses all the recipe methods on the Evaluation folder, lemmatizes them using the Spacy NLP library (more on spacy.io) and finds the most commonly used words.
The model is implemented on recipes.py and on the word_classification_model.ipynb
Afted that everything that is on the listings (ingredient, tool or technique) is assigned as such. 

### word_frequency_analysis
This is the output folder containing all the csv tables:
- word_frequency: all the found words sorted by absolute frequency
- word_frequency_[ingredients/tools/techniques]: the filtered table for those tags 

## ClassApp
Since the amount of data was not enough to train a model, we had to collect more data. 
For this I created a simple GUI app on python (Cuukin ClassApp) for users to play and classify recipe methods into their culinary techniques.
More info on this side project on githu.com/guilevieiram/classapp
The methods used were extracted from the BBC DB. 
You can find out how this is done on the "prepping_data_classapp.ipynb" notebook (on the notebooks folder)

### classapp_output
The results of this classification are saved as .db sqlite3 files in this folder.
The naming is "data_XX.db" with XX being the initials of the user (GV, RC, WZ, ...)
This tables pass to a processing before going into BERT. This consists of concatenating all recults, merging some of the columns together (e.g. all chopping related techniques goes into 'chopping+') and excluding techniques that weren't mentioned in any classification. This reduces the number of categories, bettering our results!

## Syntetic data
In order to expand this data and get better results on our classification, we tried to create alternative sentences for each recipe method without altering the meaning of the initial sentence. 

### sentence_generator(old)
This was a first attempt on generating those methods by using the NLTK library and synonyms.
The final results were very poor but the script and the notebook are in this folder if you want to try. You just need to:
from similar_sentence_generator import generate_sentences
generate_sentences(sentence, number_alternative_sentences) # returns a list of sentences of size less then number_alternative_sentences

### synthesising_data.ipynb
This is the second and final attempt on generating synthetic data.
This is the CoLab notebook that exports the generated data.
It uses PEGASUS, a transformer model developed by google for generating paraphrases of a given sentence. The code was adapted so it can support paragraphs and export everything to csvs to later use on BERT model.
More information can be found inside the notebook

## BERT multi-label classification model
This is the main part of the project. This is an adaptadion of a regular HuggingFace BERT classification transformer to handle multi-label classification.

### recipe_classification_bert.ipynb
This is the notebook used on CoLab to train the model and export it as a torch model file.
Most of the code was provided by Ronak Patel (github.com/rap12391) in this wonderful article
In short, it uses a pre-treined BERT classification model, provided by HuggingFace and uses logist functions to addapt the final layer to output probabilities. This probabilities pass through a setted threshold (0.50) to be classified as yes or no. Since these probabilities don't go under a softmax function we can have more than one value above 0.50, hence our multi label classification!
If you want to know more about transformers and bert access huggingface.co

### model
On this folder is the BERT model, downloaded from CoLab after the recipe_classification_bert.ipynb was executed.
This model is imported by the recipe.py to do the technique sorting on the given evaluation data.


# Results
As of 16th of June, we are on the data collection phase. Soon we'll be able to train the model and apply it on the BBC data base!
