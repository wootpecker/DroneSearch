import copy
import torch
import matplotlib.pyplot as plt

from models.gmrf.common.obstacle_map import ObstacleMap
from models.gmrf.common.observation import Observation
from models.gmrf.gmrf.gmrf_gas.gmrf_gas import GMRF_Gas, GMRF_Gas_Efficient


class myGMRF():
    def __init__(self, sigma_gz=None, sigma_gr=None, sigma_gb=None, gtk=None, resolution=None):
        om = ObstacleMap(dimensions=2, size=(25,30), resolution=resolution) 
        self.g = GMRF_Gas(om, 
                          sigma_gz=sigma_gz,
                          sigma_gr=sigma_gr,
                          sigma_gb=sigma_gb,
                          gtk=gtk,
                          resolution=resolution)
        
#    def calculate_old(self, y):
#        """ Calculates the distribution with GMRF. Takes true distribution as input, but grabs the positions of the sparse sensor network.
#        Must be adapted, if different sampling positions are desired."""
#        g_c = copy.deepcopy(self.g)
        
#        n = 5
#        for row in range(int(n/2), 30, n): 
#            for col in range(int(n/2), 25, n):
#                conc = y[row][col]
#                obs = Observation(position=((col,30-row)), gas=conc) # adjust the y axis
#                g_c.addObservation(obs)
#        g_c.estimate()
#        return torch.tensor(g_c.getGasEstimate()._data)
    
    def calculate(self, X):
        """ Calculates the distribution with GMRF. Must be adapted, if different sampling positions are desired."""
        g_c = copy.deepcopy(self.g)
        
        for row in range(6): 
            for col in range(5):
                conc = X[row][col]
                x_pos = col*5 + 2
                y_pos = (6-row)*5 - 2
                
                obs = Observation(position=(x_pos,y_pos), gas=conc) # adjust the y axis
                g_c.addObservation(obs)
        g_c.estimate()
        return torch.tensor(g_c.getGasEstimate()._data)


from torch.utils import data
from data.gdm_dataset import GasDataSet

dataset = GasDataSet("data/30x25/test.pt")
loader = data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
data_iter = iter(loader)






print('\n')

x_train= []#dataset[200]
y_train =[]
lenge=dataset.__len__()
print(str(lenge))
print(dataset.data)
#print(dataset.shape)
for x in dataset.data:
    print(x.size())
    if(x.size(dim=1)>5):
        x_train.append(x[0])
        #y_train.append(x[0].size())
    else:
        y_train.append(x[1])

X, y = dataset[5000]
plt.imshow(X.squeeze())


plt.imshow(y.squeeze())
plt.show()
for z in range(5):
    #plt.figure()
    plt.imshow(x_train[z].squeeze())
    f, axarr = plt.subplots(2,1) 
    axarr[0].imshow(x_train[z].squeeze())
    axarr[1].imshow(y_train[z].squeeze())
    f.suptitle('Image Nr. %z'+str(z) )
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('second plot')
    ax1.imshow(x_train[z])
    ax2.imshow(y_train[z])
    plt.show()


import numpy as np
pos = np.zeros([30,3])


np.savetxt("X.csv", pos, delimiter=",")


plt.imshow(X.squeeze())


plt.imshow(y.squeeze())




import matplotlib.pyplot as plt

DEFAULT_RES = 0.1
k = 0.8 # Correct for variance scale
gz = 0.1 / DEFAULT_RES
gr = 1.128 / DEFAULT_RES # Compensated for resolution
gb = k * 1000
DEFAULT_GTK = 0.012

gmrf = myGMRF(sigma_gz=gz, sigma_gr=gr, sigma_gb=gb, gtk=DEFAULT_GTK, resolution=1)
y_gmrf = gmrf.calculate(X.squeeze())[None,None,:]
plt.figure(figsize=(3, 2.5))
plt.imshow(y_gmrf.squeeze())
plt.axis('off');
plt.title(f"sigma_gz: {gz}, sigma_gr: {gr}, sigma_gb: {gb}")
plt.show();



