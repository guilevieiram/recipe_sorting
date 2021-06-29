import sqlite3, numpy as np

# IMPORTING ALL FUNCTIONS
from functions import evaluate_profit_over_range

def run():
    # INITIALIZING DB, NLP, TIME
    from parameters import test_db_name
    conn = sqlite3.connect(test_db_name)

    # PARAMETERS FOR OPTIMIZATION
    from parameters import number_steps, number_words, sensibility_begin, sensibility_end
    sensibility_range = np.linspace (sensibility_begin, sensibility_end, number_steps)

    # CALCULATING CM AND PROFIT
    evaluate_profit_over_range('ingredients', 'recipe_ingredients_auto', sensibility_range, number_words, conn)
    evaluate_profit_over_range('tools', 'recipe_tools_auto', sensibility_range, number_words, conn)
    evaluate_profit_over_range('techniques', 'recipe_techniques_auto', sensibility_range, number_words, conn)

    # EXITING DB
    conn.commit()
    conn.close()
