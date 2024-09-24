#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from utils import *
import multiprocessing

# Example function to compute neg2lnL for a given trmass
def neg2lnL_multiprocess(trmassvalue):
    # Replace this with your actual neg2lnL computation
    # For demonstration, we'll use a simple placeholder computation
    result = neg2lnL(trmass=trmassvalue)
    return result

def compute_neg2lnL_in_parallel(trmass_values, num_cores):
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(neg2lnL_multiprocess, trmass_values)
    return results

def save_results_to_file(filename, trmass_values, results):
    # Combine trmass values and results into a single array for saving
    data = np.column_stack((trmass_values, results))
    header = "trmass, neg2lnL"
    np.savetxt(filename, data, header=header, delimiter=',', fmt='%f')

if __name__ == "__main__":
    trmass_values = np.arange(1, 13)
    num_cores = 6
    results = compute_neg2lnL_in_parallel(trmass_values, num_cores)
    save_results_to_file('trmass_likelihood.txt', trmass_values, results)


# In[ ]:




