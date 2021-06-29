# recipe_sorting

This is a first attempt to classify recipes in techniques, ingredients and tools using their methods (text) as entries.
The core methodology is to use word2vec to compare similarity between groups of words in a given method and a technique name for example. 
The results are analysed using confusion matrices and the hyperparametrization is done considering only one variable: the sensibility treshold.
The hyperparametrization is done considering a profit model for the confusion matrix.

Final results are not the best.
The project architecture is badly made, not pythonic at all.

Im starting a new project with more robust methods and better architecture.
