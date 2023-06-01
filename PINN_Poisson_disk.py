from torch import nn


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_style("darkgrid")

from time import time

from base_MLP import base_MLP


import torch
from torch import nn
from torchview import draw_graph
import torchinfo 

class base_MLP(nn.Module):
    ''' Definition of a baseline MLP model '''

    def __init__(self,nun_hidden_layers=1,in_features=10,hidden_features=10,out_features=1,device=None):
        super().__init__()

        self.input_size = (1,in_features)
        self.layer_in = nn.Linear(in_features, hidden_features)
        self.layers_hid = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for i in range(nun_hidden_layers-1)])
        self.layer_out = nn.Linear(hidden_features, out_features)

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device);

    def forward(self,x):
        x = self.layer_in(x)
        for layer in self.layers_hid:
            x = layer(x)
        return self.layer_out(x)


    # -----------------------------------------------
    def summary(self):
        return torchinfo.summary(self, input_size=self.input_size,device=self.device)
        
    
    # -----------------------------------------------
    def visual_graph(self):
        model_graph = draw_graph(self, 
                                 input_size=self.input_size, 
                                 expand_nested=True, 
                                 device=self.device)
        return model_graph.visual_graph
    



model = base_MLP(nun_hidden_layers=5)

model.visual_graph()
model.summary()

# model.forward(torch.from_numpy(np.arange(10)).to(model.device))

###
rr = np.linspace(0,1,50)
theta = np.linspace(0,2*np.pi,100)

RR,THETA = np.meshgrid(rr,theta)
xx,yy = RR*np.cos(THETA),RR*np.sin(THETA)
f = 1 # source term
def f_ref(x,y,f):
    return .25*(np.power(f,2)-np.power(x,2)-np.power(y,2))

plt.figure()
plt.contourf(xx,yy,f_ref(xx,yy,f),30)
plt.axis('equal')
plt.colorbar()
plt.title('Reference solution')
plt.show()
















