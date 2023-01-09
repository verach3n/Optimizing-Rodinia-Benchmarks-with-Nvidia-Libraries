import subprocess
import math
# dictionary to store the results
results = {}

# loop through the sizes
for size in [11, 11.5 , 12, 12.5, 13, 13.5, 14, 14.2]:
  # store the sum of the running times for this size
  print('Running size', size)
  total_time = 0
  
  # run the executable 10 times for this size
  for i in range(10):
    # print('Running size', size, 'iteration', i)
    # run the executable and get the output
    output = subprocess.run(['./build/cusolver_getrf_example', str(math.ceil((2**size)))], capture_output=True)
    # the output will be in bytes, so we decode it to get the string
    output_str = output.stdout.decode('utf-8')
    # the output is a string representation of a floating point number, so we convert it to a float
    running_time = float(output_str)
    # add the running time to the total
    total_time += running_time
  
  # calculate the average running time for this size
  average_time = total_time / 10
  # store the average running time in the results dictionary
  results[str(size)] = average_time

# print the results
print('times = [')
for k, v in results.items():
  print(v, ",")
print(']')

print(results)
