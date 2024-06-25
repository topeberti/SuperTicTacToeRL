from joblib import Parallel, delayed

# Define the function you want to parallelize
def my_function(i):
    return i ** 2

# Create an instance of Parallel and use delayed to specify the function to parallelize
results = Parallel(n_jobs=-1)(delayed(my_function)(i) for i in range(10))

print(results)