test_run = True

test_db_name = 'recipes_test.db' # for testing, use ':memory:' to create DB in RAM
run_db_name = 'recipes.db'

directory = r"C:\Users\Bruna Fernandes\Desktop\gui\Recipe_sorting"
CM_plot_folder = "\CM_plots"
optimization_plot_folder = "\optimization_plots"
test_csv_folder = "\\test_csv"
run_csv_folder = "\\run_csv"

number_steps = 20
number_words = 2
sensibility_begin =  0.5
sensibility_end = 1

number_words_ingredients = 2
number_words_techniques = 2
number_words_tools = 2

TP_cost = 2
FP_cost = -5
FN_cost = -1
TN_cost = 1

positive_weight = 0.7
negative_weight = 0.3

CM_size = (3,2)

# HANDMADE SORTING_PARAMETERS
sensibility_ingredients = 0.957
sensibility_tools = 0.831
sensibility_techniques = 1
# sorting_parameters = [sensibility_ingredients, sensibility_tools, sensibility_techniques]

# AUTOMATIC GENERATED PARAMETERS (MUST BE ON THE LAST LINE OF CODE)
sorting_parameters = [0.9736842105263157, 0.763157894736842, 1.0]