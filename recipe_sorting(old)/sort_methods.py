import sqlite3

# IMPORTING FUNCTIONS
from functions import write_csv_from_db
from functions import sort_recipes_methods

def run():
    # TEST RUN OR NOT
    from parameters import test_run

    if test_run == True:
        from parameters import test_db_name
        conn = sqlite3.connect(test_db_name)
        
    if test_run == False:
        from parameters import run_db_name
        conn = sqlite3.connect(run_db_name)

    # DEFINING SORTING PARAMETERS
    from parameters import number_words_ingredients, number_words_techniques, number_words_tools, sorting_parameters
    [sensibility_ingredients, sensibility_techniques, sensibility_tools] = sorting_parameters

    # SORTING METHODS
    sort_recipes_methods('ingredients', 'recipe_ingredients_auto', sensibility_ingredients, number_words_ingredients, conn)
    sort_recipes_methods('tools', 'recipe_tools_auto', sensibility_tools, number_words_tools, conn)
    sort_recipes_methods('techniques', 'recipe_techniques_auto', sensibility_techniques, number_words_techniques, conn)

    # WRITING CSV FILES
    write_csv_from_db('recipe_ingredients_auto', conn)
    write_csv_from_db('recipe_tools_auto', conn)
    write_csv_from_db('recipe_techniques_auto', conn)

    # EXITING DB
    conn.commit()
    conn.close()
