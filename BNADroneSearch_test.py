import unittest
import numpy as np
import BNADroneSearch as BNA
class TestStringMethods(unittest.TestCase):



    def matrix_helper_function(arr):
        zeros=np.zeros((6,5))
        arr[0]

    def create_matrix(self,dims):
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
        return np.array(all_cases)



    def test_transform_into_6x5(self):
        dims=[30,25]
        all_cases=self.create_matrix(dims)
        reduced_dims=[6,5]
        reduced_cases=self.create_matrix(reduced_dims)

        transform_function=BNA.transform_into_6x5(all_cases)
        np.testing.assert_array_equal(transform_function,reduced_cases)



    def test_transform_into_6x5_multiple(self):
        all_cases=[]
        dims=[30,25]
        for i in range(dims[0]):
            for j in range(dims[1]):
                zeros=np.zeros((dims[0],dims[1]))
                zeros[i,j]=1
                zeros[0,0]=1
                all_cases.append(zeros)
        all_cases=np.array(all_cases)
        transform_function=BNA.transform_into_6x5(all_cases)
        


if __name__ == '__main__':
    unittest.main()