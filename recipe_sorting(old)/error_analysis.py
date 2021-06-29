import sqlite3

# IMPORTING FUNCTIONS
from functions import calculate_CM, profit, plot_CM, relative_CM

def run():
    # INITIALIZING DB
    from parameters import test_db_name
    conn = sqlite3.connect(test_db_name)

    # CALCULATING CONFUSION MATRICES
    CM_ingredients = calculate_CM('ingredients', conn)
    CM_tools = calculate_CM('tools', conn)
    CM_techniques = calculate_CM('techniques', conn)

    # CALCULATING RELATIVE CONFUSION MATRICES
    RCM_ingredients = relative_CM(CM_ingredients)
    RCM_tools = relative_CM(CM_tools)
    RCM_techniques = relative_CM(CM_techniques)

    # PLOTTING THE MATRICES
    plot_CM('ingredients', RCM_ingredients)
    plot_CM('tools', RCM_tools)
    plot_CM('techniques', RCM_techniques)

    # EXITING DB
    conn.commit()
    conn.close()
