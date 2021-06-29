import sqlite3

# IMPORTING FUNCTIONS
from functions import  insert_into_db

def run():  
    # TEST RUN OR NOT
    from parameters import test_run
    
    if test_run == True:
        from parameters import test_db_name
        conn = sqlite3.connect(test_db_name)
        # IMPORTING VALIDATION CSV TABLES INTO DB
        insert_into_db('recipe_tools', 'recipe-tools.csv', conn)
        insert_into_db('recipe_techniques', 'recipe-techniques.csv', conn)
        
    if test_run == False:
        from parameters import run_db_name
        conn = sqlite3.connect(run_db_name)


    # IMPORTING REST OF CSV TABLES INTO DB
    insert_into_db('recipes', 'recipes.csv', conn)

    insert_into_db('recipe_methods', 'recipe-methods.csv', conn)
    insert_into_db('recipe_ingredients', 'recipe-ingredients.csv', conn)
    
    insert_into_db('ingredients', 'ingredients.csv', conn)
    insert_into_db('tools', 'tools.csv', conn)
    insert_into_db('techniques', 'techniques.csv', conn)


    # EXITING DB
    conn.commit()
    conn.close()

        