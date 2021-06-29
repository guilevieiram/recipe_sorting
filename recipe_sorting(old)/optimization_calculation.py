import sqlite3

# IMPORTING FUNCTIONS
from functions import plot_CM_cell, find_maximum, export_sorting_parameters

def run():
    # INITIALIZING DB
    from parameters import test_db_name
    conn = sqlite3.connect(test_db_name)

    # PARAMETERS FOR PLOTTING
    x_axis = "sensibility"

    # PLOTTING ERROR GRAPHS
    y_axis = "TP"
    plot_CM_cell(x_axis, y_axis, conn)
    y_axis = "FP"
    plot_CM_cell(x_axis, y_axis, conn)
    y_axis = "FN"
    plot_CM_cell(x_axis, y_axis, conn)
    y_axis = "TN"
    plot_CM_cell(x_axis, y_axis, conn)
    y_axis = "profit"
    plot_CM_cell(x_axis, y_axis, conn)


    # FINDING MAXIMUM OF PROFIT FUNCTIONS
    [ingredients_sensibility, ingredients_maximum_profit] = find_maximum(x_axis, y_axis, "ingredients", conn)
    [tools_sensibility, tools_maximum_profit] = find_maximum(x_axis, y_axis, "tools", conn)
    [techniques_sensibility, techniques_maximum_profit] = find_maximum(x_axis, y_axis, "techniques", conn)

    # PRINTING SENSIBILITY RESULTS

    print("\nIDEAL INGREDIENTS SENSIBILITY: ", ingredients_sensibility)
    print("\nIDEAL TOOLS SENSIBILITY: ", tools_sensibility)
    print("\nIDEAL TECHNIQUES SENSIBILITY: ", techniques_sensibility)


    # WRITING SORTING PARAMETERS IN THE PARAMETER SCRIPT
    parameters = [ingredients_sensibility, tools_sensibility, techniques_sensibility]
    export_sorting_parameters(parameters)
   
    
    # EXITING DB
    conn.commit()
    conn.close()
    

