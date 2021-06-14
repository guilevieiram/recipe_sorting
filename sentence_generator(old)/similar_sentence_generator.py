# Importing modules
import nltk 
import pandas as pd

## Downloading nltk submodules if not in machine
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Importing word bank and functions
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer
lemmatizer = WordNetLemmatizer()

# Main function
def generate_sentences(root_sentence, number_sentences, similarity_treshold = 0.5):
    
    '''
    This functions generates similar sentences given a root/parent sentence.

    It does this by searching for synonyms for each word in the sentence
    and selecting them based on the given word funcion (noun, verb, ...) and the similarity
    between the two words.

    This function holds two parameters: 
    - number_sentences: maximum number of sentences to be generated
    - similarity_treshold: minimum value of similarity for a synonym to be considered

    The sentences are returned as strings in a list.


    BUG: sometimes the same sentence is generated, which is not good

    '''
    # Tokenizing and tagging root sentence
    tokenized = word_tokenize(root_sentence)
    tagged_proper = synonym_tagger(tokenized)

    # Initializing synonym dictionary
    syn_dict = {}
    for word, tag in tagged_proper: syn_dict[word] = []

    # Filling synonym dictionary
    for word_tag in tagged_proper:
        find_synonyms(word_tag, syn_dict, similarity_treshold)

    # Converting synonyms dict to data frame
    synonyms_df = dict_to_data_frame(syn_dict, tagged_proper)
    
    # Initializing sentences list parameters
    sentences = []
    sentence_counter = 0
    number_synonyms = synonyms_df.shape[0]

    # Main loop for creating sentences
    for index in range(number_synonyms):

        # grabbing word and synonym
        word = synonyms_df.at[index, 'word']
        syn = synonyms_df.at[index, 'syn']

        # making copy of sentence and change word for synonym
        word_index = tokenized.index(word)
        sent = [t for t in tokenized]
        # this replace is made to handle more that one word synonym
        sent[word_index] = syn.replace('_', ' ') 

        # transforming tokenized sentence into string
        untokenized_sentence = TreebankWordDetokenizer().detokenize(sent)

        # appending string into list 
        sentences.append(untokenized_sentence)
        sentence_counter += 1

        #checks if the number of sentences desired was fulfilled
        if sentence_counter >= number_sentences: break

    return sentences


# Auxiliary functions
def find_synonyms(word_tagged, dict, treshold):

    '''
    This function finds synonyms of words that meet a certain criteria.

    The words are passed in the "word_tagged" parameter as (word, tag) tuple with
    this tag being 'n', 'v', 'r' or 'a' for noun, verb, adverb, and adjective.

    The similarities and synonyms are calculated using the Wordnet from nltk.

    '''

    # Separate word and tag and lemmatizes word
    word, tag = word_tagged
    lemma = lemmatizer.lemmatize(word)

    # Main loop
    try:
        # Convert word to wordnet syn format
        word_syn = wordnet.synset(lemma + '.' + tag + '.01')

        # Loops through all synonyms with the same tag
        for syn in wordnet.synsets(word, wordnet_tag(tag)):
            
            # Calculates similarity
            similarity = word_syn.wup_similarity(syn)

            # Checks treshold condition
            if similarity < treshold: break

            # Append lemmas and similarity values in the dict as tuples
            for lemma in syn.lemmas():
                if lemma.name() not in [x[0] for x in dict[word]]: 
                    dict[word].append((lemma.name(), similarity))
    except:
        pass

def synonym_tagger (words):

    '''
    This function tags word (in the form of list of tokens) in a simpler way.

    It starts by tagging in the regular NLTK format and then reduces it to 
    account only for nouns, verbs and adverbs.

    For some reason, the adjectives as 'a' are not supported in other nltk syn functions
    so I just took that classification out.

    '''

    # Tags the words as expected
    tagged = pos_tag(words)

    # Initializes result list. 
    results = []

    # Appends result list with tuple (word, tag), where tag is a simple version
    for word, tag in tagged:
        if tag.startswith('N'): results.append((word, 'n'))
        if tag.startswith('V'): results.append((word, 'v'))
        if tag.startswith('R'): results.append((word, 'r'))
        # if tag.startswith('JJ'): results.append((word, 'a'))
    
    # Return the resulting list
    return results

def wordnet_tag(tag):
    '''

    This function uses the simple tag calculated in "synonym tagger" 
    to create wordnet tag objects. Those objects are used in the "synsets" function
    later.

    Again, I cant figure out why adjectives are not supported by those functions.

    '''

    if tag == 'n': return wordnet.NOUN
    if tag == 'v': return wordnet.VERB
    if tag == 'r': return wordnet.ADV
    # if tag == 'a': return wordnet.ADJ

def dict_to_data_frame(synonym_dict, tagged):
    
    '''

    This function creates a data frame from the synonym dictionary provided.

    This data frame has the form (word, synonym, simple tag, similarity value)
    for all the words provided in the dictionary keys.

    The tag column is separated in 3 other columns with binary values for simplicity.

    Then this table is sorted in a tag preference order and, after, in the similarity value
    This is done do guarantee the best results in the first rows

    The tag preference order is done in a way that the meaning of the sentence is less altered
    in the first synonyms proposed and more altered in the last.

    '''

    # Create empty data frame by columns 
    df = pd.DataFrame(columns=['word', 'syn', 'tag', 'sim'])

    # Append rows with desired values
    for word in synonym_dict.keys():
        for syn in synonym_dict[word]:
            row = {
                'word': word, 
                'syn': syn[0],
                'tag': dict(tagged)[word],
                'sim': syn[1]
            }
            df = df.append(row, ignore_index=True)

    # Separates the tag column in 3 binary columns
    df = pd.get_dummies(df, columns=['tag'])

    # Check if there are missing columns and create them
    tags = ['tag_n', 'tag_v', 'tag_r', 'tag_a']
    for tag in tags: 
        if tag not in df: df[tag] = 0

    # Sets tag synonym preference
    simple_tags = ['r', 'n', 'v']
    substitution_order_preference = ["tag_" + tag for tag in simple_tags]
    

    # Sort data frame
    df.sort_values(substitution_order_preference + ['sim'], ascending=False,  ignore_index=True, inplace=True)

    # Removing word and synonym equal values
    df = df[df['word'] != df['syn']].reset_index()

    # Returning data frame
    return df