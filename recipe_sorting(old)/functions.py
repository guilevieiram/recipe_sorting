import sqlite3, csv, spacy, time
import pandas as pd, matplotlib.pyplot as plt, numpy as np, seaborn as sn

nlp = spacy.load("en_core_web_lg")

# SORTING MODELS 

def sort_recipes_methods(type_of_skill, table_name, limit_score_value, n_words, connection):
    if type_of_skill not in ["ingredients", "techniques", "tools"]: print("error: wrong cathegory name\n")

    #CREATING TABLE
    c = connection.cursor()
    header_cathegories = ("method_id", "method_index", "recipe_id", type_of_skill + "_name", "score")
    header = ','.join(header_cathegories)
    sql = "CREATE TABLE " + table_name + " (" + header + ")"
    c.execute(sql)
    connection.commit()

    #FETCHING METHODS TABLE
    sql = "SELECT * FROM recipe_methods"
    c.execute(sql)
    table_methods = c.fetchall()
    connection.commit()

    #FETCHING DESIRED TABLE
    sql = "SELECT * FROM " + type_of_skill
    c.execute(sql)
    skill_list = c.fetchall()
    connection.commit()

    #CHECKING SIMMILARITY
    for methods_row in table_methods:
        #CREATING A METHOD DOCUMENT AND LEMMATIZING IT
        method_description_string = methods_row[2]
        method_id = methods_row[0]
        method_index = methods_row[6]
        recipe_index = methods_row[7]
        method_document = lemmatize(method_description_string)
        
        for skill in skill_list:
            #CREATING A SKILL DOCUMENT AND LEMMATIZING IT
            skill_name_string = skill[1]
            skill_document = lemmatize(skill_name_string)
            score = 0

            #CALCULATING SCORE FOR EACH SKILL
            if len(skill_document) != 0 and len(method_document) != 0:
                score = max(
                    skill_document.similarity(method_document[count:count+n_words]) 
                    for count in range(len(method_document)-(n_words-1))
                )
            
            #CHECKING THRESHOLD
            if score > limit_score_value:

                #INSERTING INTO TABLE
                place_holder_string = ','.join("?" for count in range(len(header_cathegories)))
                values_to_insert = (method_id, method_index, recipe_index, skill_name_string, score) 

                sql = "INSERT INTO " + table_name + " VALUES (" + place_holder_string + ")"
                c.execute(sql, values_to_insert)
                connection.commit()
    connection.commit()



# MAIN FUNCTIONS

def insert_into_db(table_name, csv_file, connection):
    c = connection.cursor()

    # TEST RUN OR NOT
    from parameters import test_run 
    
    if test_run == True:
        from parameters import directory, test_csv_folder, test_db_name
        conn = sqlite3.connect(test_db_name)
        path = directory + test_csv_folder + "\\"
    
    if test_run == False:
        from parameters import directory, run_csv_folder, run_db_name
        conn = sqlite3.connect(run_db_name)
        path = directory + run_csv_folder + "\\"

    csv_file = path + csv_file

    with open (csv_file, 'r', encoding="utf-8") as file:
        table = csv.reader(file, quotechar='"', delimiter = ',')
        header = next(table)
        num_columns = len(header)
        joined_header = ','.join(header)
        
        sql = "CREATE TABLE " + table_name + " (" + joined_header + ")"
        
        c.execute(sql)

        place_holder_string = ','.join("?" for i in range(num_columns))

        sql = "INSERT INTO " + table_name + " VALUES (" + place_holder_string + ")"

        for row in table:
            c.execute(sql, row)
            connection.commit()

    connection.commit()

def calculate_CM (type_of_skill, connection):
    if type_of_skill not in ['ingredients', 'tools', 'techniques']: print("wrong cathegory")    
    
    c = connection.cursor()

    # INITIATING CM CELLS
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # GETTING FILLED RECIPES ID
    sql = """ 
    SELECT Id FROM recipes
     """
    c.execute(sql)
    all_recipes_id = c.fetchall()
    all_recipes_id = normalize_query_vector(all_recipes_id)

    # GETTING LIST OF THAT SKILL
    sql = "SELECT name FROM " + type_of_skill
    c.execute(sql)
    skill_list = c.fetchall()
    skill_list = normalize_query_vector(skill_list)

    for recipe_id in all_recipes_id:
        
        # SELECTING ACTUAL AND PREDICTED SKILL LISTS ASSOCIATED TO A RECIPE
        sql = " SELECT DISTINCT " + type_of_skill + "_name"  
        sql += " FROM recipe_" + type_of_skill + "_auto"
        sql += " WHERE recipe_id = " + str(recipe_id)
        sql += " OR recipe_id = '" + str(recipe_id) +"'"
        
        c.execute(sql)
        list_predicted = c.fetchall()
        list_predicted = normalize_query_vector(list_predicted)

        sql = " SELECT DISTINCT " + type_of_skill + "_name"  
        sql += " FROM recipe_" + type_of_skill
        sql += " WHERE recipe_id = " + str(recipe_id)
        sql += " OR recipe_id = '" + str(recipe_id) +"'"

        c.execute(sql)
        list_actual = c.fetchall()
        list_actual = normalize_query_vector(list_actual)
        connection.commit()

        # CALCULATING CM CELLS
        for skill in skill_list:
            if skill in list_predicted and skill in list_actual: TP += 1
            if skill in list_predicted and skill not in list_actual: FP += 1
            if skill not in list_predicted and skill in list_actual: FN += 1
            if skill not in list_predicted and skill not in list_actual: TN += 1

    confusion_matrix = [[TP, FP],[FN, TN]]
    return confusion_matrix

def evaluate_profit_over_range(type_of_skill, table_name, sensibility_range, number_words, connection):
    # PARAMETERS
    index = 0
    c = connection.cursor()

    # DELETING TABLE IF IT EXISTS
    sql = "SELECT count(name) FROM sqlite_master WHERE type = 'table' AND name= '"+ table_name + "'"
    c.execute(sql)
    if c.fetchone()[0] == 1:
        sql = "DROP TABLE " + table_name
        c.execute(sql)
        connection.commit()
    
    # CREATING PROFIT DB TABLE
    CM_table_name = type_of_skill + "_CM"
    header = ["sensibility", "TP", "FP", "FN", "TN", "profit"]
    joined_header = ",".join(header)
    sql = " CREATE TABLE " + CM_table_name + " (" + joined_header + ")"
    c.execute(sql)
    connection.commit()

    # EVALUATING ERRORS IN SENSIBILITY RANGE AND INSERTING IN TABLE
    for sensibility in sensibility_range:
        print("\tStep ", index + 1 , " out of ", len(sensibility_range))

        sort_recipes_methods(type_of_skill, table_name, sensibility, number_words, connection)
        CM = relative_CM(calculate_CM(type_of_skill, connection))
        TP = CM[0][0] 
        FP = CM[0][1] 
        FN = CM[1][0] 
        TN = CM[1][1] 
        profit_value = profit(CM)
        sql = "DROP TABLE " + table_name
        c.execute(sql)
        connection.commit()

        values = [sensibility]
        values.append(TP)
        values.append(FP)
        values.append(FN)
        values.append(TN)
        values.append(profit_value)
        
        num_columns = len(values)
        values_holder = ','.join("?" for count in range(num_columns))

        sql = "INSERT INTO " + CM_table_name + " VALUES (" + values_holder + ")"
        c.execute(sql,values)
        connection.commit()

        index += 1

# SECONDARY FUNCTIONS

def column (matrix, column_number):
    return [row[column_number] for row in matrix ]

def find_maximum(x_axis, y_axis, type_of_skill, connection):

    c = connection.cursor()

    sql = "SELECT " + x_axis + ", " + y_axis
    sql += " FROM " + type_of_skill + "_CM"

    c.execute(sql)
    points = c.fetchall()
    connection.commit()

    x_values = column(points, 0)
    y_values = column(points, 1)
    
    y_maximum = max(y_values)
    index = y_values.index(y_maximum)
    x_maximum = x_values[index]

    return [x_maximum, y_maximum]

def relative_CM(confusion_matrix):
    TP = confusion_matrix[0][0] 
    FP = confusion_matrix[0][1] 
    FN = confusion_matrix[1][0] 
    TN = confusion_matrix[1][1] 
    if (TP + FP) != 0 and (TN + FN) !=0:
        RTP = TP / (TP + FP)
        RFP = FP / (TP + FP)
        RFN = FN / (TN + FN)
        RTN = TN / (TN + FN)
    else: RTP = RFP = RFN = RTN = 0
    
    return [[RTP, RFP], [RFN, RTN]]

def plot_CM(type_of_skill, confusion_matrix):
    index_labels = ["YES", "NO"]
    column_labels = ["YES", "NO"]
    plot_name = "CM_" + type_of_skill
    df_cm = pd.DataFrame(confusion_matrix, index = index_labels, columns = column_labels)

    from parameters import CM_size
    plt.figure(figsize = CM_size)
    sn.heatmap(df_cm, annot=True)

    from parameters import directory, CM_plot_folder
    path = directory + CM_plot_folder
    plt.savefig(path + "\\" + plot_name)  

def normalize_query_vector(vector):
    for index in range(len(vector)): 
        element = vector[index][0]
        if element.isnumeric(): vector[index] = int(element)
        else: vector[index] = element

    return vector

def write_csv_from_db(table_name, connection):
    from parameters import test_run

    if test_run == True:
        from parameters import directory, test_csv_folder, test_db_name
        conn = sqlite3.connect(test_db_name)
        path = directory + test_csv_folder + "\\"
    
    if test_run == False:
        from parameters import directory, run_csv_folder, run_db_name
        conn = sqlite3.connect(run_db_name)
        path = directory + run_csv_folder + "\\"
    
    sql = "SELECT * FROM " + table_name

    csv_name = path + table_name + ".csv"
    dataframe = pd.read_sql_query(sql, connection)
    dataframe.to_csv(csv_name, index=False)

def lemmatize (input_string):
    doc = nlp(input_string)
    lemma_string = " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
    lemma_doc = nlp(lemma_string)
    return lemma_doc

def profit(confusion_matrix):
    TP = confusion_matrix[0][0] 
    FP = confusion_matrix[0][1] 
    FN = confusion_matrix[1][0] 
    TN = confusion_matrix[1][1] 

    from parameters import TP_cost, FP_cost, FN_cost, TN_cost

    profit = TP * TP_cost + FP * FP_cost + FN * FN_cost + TN * TN_cost
    return profit

def plot_CM_cell(x_axis, y_axis, connection):
    plt.clf()
    type_of_skill = ["ingredients", "tools", "techniques"]  
    plot_name =  y_axis + ".png"
    c = connection.cursor()

    for cathegory in type_of_skill:

        sql = "SELECT " + x_axis + ", " + y_axis
        sql += " FROM " + cathegory + "_CM"

        c.execute(sql)
        points = c.fetchall()
        connection.commit()

        x = column(points, 0)
        y = column(points, 1)

        plt.plot(x, y, label = cathegory)
        
    plt.xlabel = x_axis
    plt.ylabel = y_axis
    plt.legend()
    
    from parameters  import directory, optimization_plot_folder
    path = directory + optimization_plot_folder

    plt.savefig(path + "\\" + plot_name)  

def export_sorting_parameters(parameters):

    from parameters import directory
    directory += "\source\parameters.py"

    parameters_string_list = []
    for parameter in parameters:
        parameters_string_list.append(str(parameter))

    parameters_string = "sorting_parameters = [" + ", ".join(parameters_string_list) + "]"

    with open(directory, "r") as file_handle:
        # CHANGES LAST LINE TO sorting_parameters
        list_lines = file_handle.readlines()
        list_lines[len(list_lines)-1] = parameters_string
        
    with open(directory, "w") as file_handle:
        file_handle.writelines(list_lines)

# UNUSED/PAST FUNCTIONS

def find_minimum(x_axis, y_axis, type_of_skill, connection):

    c = connection.cursor()

    sql = "SELECT " + x_axis + ", " + y_axis
    sql += " FROM " + type_of_skill + "_errors"

    c.execute(sql)
    points = c.fetchall()
    connection.commit()

    x_values = column(points, 0)
    y_values = column(points, 1)
    
    y_minimum = min(y_values)
    index = y_values.index(y_minimum)
    x_minimum = x_values[index]

    return [x_minimum, y_minimum]

def evaluate_errors(type_of_skill, table_name, sensibility_range, number_words, connection):
    # PARAMETERS
    errors = []
    index = 0
    c = connection.cursor()

    # DELETING TABLE IF IT EXISTS
    sql = "SELECT count(name) FROM sqlite_master WHERE type = 'table' AND name= '"+ table_name + "'"
    c.execute(sql)
    if c.fetchone()[0] == 1:
        sql = "DROP TABLE " + table_name
        c.execute(sql)
        connection.commit()
    
    # CREATING ERRORS DB TABLE
    errors_table_name = type_of_skill + "_errors"
    header = ["sensibility", "error_positive", "error_negative", "error_total"]
    header += ["error_relative_positive", "error_relative_negative", "error_relative_total"]
    joined_header = ",".join(header)
    sql = " CREATE TABLE " + errors_table_name + " (" + joined_header + ")"
    c.execute(sql)
    connection.commit()

    # EVALUATING ERRORS IN SENSIBILITY RANGE AND INSERTING IN TABLE
    for sensibility in sensibility_range:
        print("\tStep ", index + 1 , " out of ", len(sensibility_range))

        sort_recipes_methods(type_of_skill, table_name, sensibility, number_words, connection)
        errors = counting_errors(type_of_skill, connection)
        sql = "DROP TABLE " + table_name
        c.execute(sql)
        connection.commit()

        values = [sensibility]
        for error in errors: values.append(error)
        num_columns = len(values)
        values_holder = ','.join("?" for count in range(num_columns))

        sql = "INSERT INTO " + errors_table_name + " VALUES (" + values_holder + ")"
        c.execute(sql,values)
        connection.commit()

        index += 1

def counting_errors(type_of_skill, connection): 
    if type_of_skill not in ['ingredients', 'tools', 'techniques']: return [0,0,0,0,0,0]

    c = connection.cursor()

    # DEFINING ERRORS
    error_positive = 0 # auto generated skills not found in manual set
    error_negative = 0 # manual skills not found in automatic generated set
    error_total = 0
    error_relative_positive = 0
    error_relative_negative = 0
    error_relative_total = 0
    number_of_skills = 0

 
    # GETTING FILLED RECIPES ID
    sql = """ 
    SELECT Id FROM recipes
     """
    c.execute(sql)
    all_recipes_id = c.fetchall()
    all_recipes_id = normalize_query_vector(all_recipes_id)

    for recipe_id in all_recipes_id:
        
        # SELECTING AUTOMATIC AND MANUAL SKILL LISTS ASSOCIATED TO A RECIPE
        sql = " SELECT DISTINCT " + type_of_skill + "_name"  
        sql += " FROM recipe_" + type_of_skill + "_auto"
        sql += " WHERE recipe_id = " + str(recipe_id)
        sql += " OR recipe_id = '" + str(recipe_id) +"'"
        
        c.execute(sql)
        list_auto = c.fetchall()

        sql = " SELECT DISTINCT " + type_of_skill + "_name"  
        sql += " FROM recipe_" + type_of_skill
        sql += " WHERE recipe_id = " + str(recipe_id)
        sql += " OR recipe_id = '" + str(recipe_id) +"'"

        c.execute(sql)
        list_manual = c.fetchall()
        connection.commit()

        # NORMALIZING SKILLS LISTS
        list_auto = normalize_query_vector(list_auto)
        list_manual = normalize_query_vector(list_manual)
        number_of_skills += len(list_manual)
        
        # COUNTING ERRORS
        for skill_auto in list_auto:
            if skill_auto not in list_manual: error_positive +=1

        for skill_manual in list_manual:
            if skill_manual not in list_auto: error_negative +=1     

    #CALCULATING AND RETURNING ERROR VECTOR
    error_total = error_negative + error_positive

    if number_of_skills != 0:
        error_relative_positive = error_positive / number_of_skills
        error_relative_negative = error_negative / number_of_skills

    from parameters import negative_weight, positive_weight
    error_relative_total = (positive_weight * error_relative_positive + negative_weight * error_relative_negative)/(positive_weight + negative_weight)
    error_vector = [error_positive, error_negative, error_total, error_relative_positive, error_relative_negative, error_relative_total] 

    return error_vector


