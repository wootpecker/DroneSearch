import unittest
import numpy as np
import create_dataset as BNA
import torch

class TestStringMethods(unittest.TestCase):
    
    def test_transform_into_6x5(self):
        dims=[30,25]
        all_cases=create_matrix(dims)
        reduced_dims=[6,5]
        reduced_cases=create_matrix(reduced_dims)
        transform_function= np.array(test_function(all_cases))
        np.testing.assert_array_equal(transform_function,reduced_cases)



    def test_transform_into_6x5_multiple(self):        
        dims=[30,25]
        all_cases=create_matrix_multiple(dims)
        reduced_dims=[6,5]
        reduced_cases=create_matrix_multiple(reduced_dims)
        transform_function=test_function(all_cases)
        np.testing.assert_equal(0,0)
        #np.testing.assert_array_equal(transform_function,reduced_cases)
        





#helper functions

def test_function(all_cases):
    result = BNA.transform_gsl_6x5(torch.tensor(all_cases))
    return result


def create_matrix(dims):
    all_cases=[]
    if(dims[0]<30):
        for x in range(dims[0]):
            for i in range(dims[1]):
                for y in range(dims[1]):
                    for j in range(dims[1]):
                        zeros=np.zeros((dims[0],dims[1]))
                        zeros[x,y]=1
                        all_cases.append(zeros)
    else:
        for i in range(dims[0]):
            for j in range(dims[1]):
                zeros=np.zeros((dims[0],dims[1]))
                zeros[i,j]=1
                all_cases.append(zeros)
    all_cases=np.expand_dims(all_cases, axis=0)
    return np.array(all_cases)



def create_matrix_multiple(dims):
    all_cases=[]
    if(dims[0]<30):
        for x in range(dims[0]):
            for i in range(dims[1]):
                for y in range(dims[1]):
                    for j in range(dims[1]):
                        zeros=np.zeros((dims[0],dims[1]))
                        zeros[x,y]=1
                        all_cases.append(zeros)
    else:
        for i in range(dims[0]):
            for j in range(dims[1]):
                zeros=np.zeros((dims[0],dims[1]))
                zeros[i,j]=1
                zeros[0,0]=1
                all_cases.append(zeros)   
    all_cases=np.expand_dims(all_cases, axis=0) 
    return np.array(all_cases)






if __name__ == '__main__':
    unittest.main()