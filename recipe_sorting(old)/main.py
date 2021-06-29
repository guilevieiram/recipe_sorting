import time
import create_db, sensibility_optimization, optimization_calculation, sort_methods, error_analysis

# CHECKING EXECUTION TIME
start_time = time.time()

# TEST RUN OR NOT 
from parameters import test_run

if test_run == True:
    create_db.run()
    sensibility_optimization.run()
    optimization_calculation.run()
    sort_methods.run()
    error_analysis.run()  

if test_run == False:
    create_db.run()
    sort_methods.run()
    
# CHECKING EXECUTION TIME
end_time = time.time()
print("\n\nSUCCESS!\n\n total execution time = ", end_time - start_time, " seconds")
