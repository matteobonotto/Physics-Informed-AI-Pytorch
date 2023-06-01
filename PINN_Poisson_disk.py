from torch import nn


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import path
from torch.utils.data import DataLoader
import torch.optim as optim
import seaborn as sns
sns.set_style("darkgrid")

import time 

import torch


from base_MLP import base_MLP



model = base_MLP(nun_hidden_layers=5,
                 in_features=10,
                 hidden_features=10)

model.visual_graph()
model.summary()



### Define PDE problem
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


### Define Collocation points
rr = 1
theta = np.linspace(0,2*np.pi,100)
boundary = np.column_stack((rr*np.cos(theta),rr*np.sin(theta)))
p = path.Path(boundary)  # square with legs length 1 and bottom left corner at the origin

scaler = MinMaxScaler(feature_range=(-1, 1))
N_rand_1 = 20
N_rand = round(4/np.pi*N_rand_1**2)

coll_pts = np.column_stack((scaler.fit_transform(np.random.rand(N_rand).reshape(-1,1)),
                          scaler.fit_transform(np.random.rand(N_rand).reshape(-1,1))))

coll_pts = coll_pts[p.contains_points(coll_pts),:]
# abs(100*(coll_pts.shape[0]-N_rand_1**2))/N_rand_1**2

plt.figure()
plt.plot(boundary[:,0],boundary[:,1])
plt.scatter(coll_pts[:,0],coll_pts[:,1], c='r', marker='*', alpha=0.2)
plt.axis('equal')
plt.title('N = ' + str(coll_pts.shape[0]) + ' collocation points')
plt.show()




###
rr = 1
theta = np.linspace(0,2*np.pi,200)
# X_train = np.column_stack((rr*np.cos(theta),rr*np.sin(theta)))

X_train = coll_pts
Y_train = f_ref(X_train[:,0],X_train[:,1],f).reshape(-1,1)

# X_r    = torch.from_numpy(coll_pts).float()  # -> collocation points
X_train = torch.from_numpy(X_train).float()   # -> boundary points
Y_train = torch.from_numpy(Y_train).float()  #-> values of function at boundary points

n_input = 2
n_output = 1




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





class trainer(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.device = model.device

    # -----------------------------------------------
    def data_loader(self, data, batch_size, num_workers=0, pin_memory=True, shuffle=True):
        return DataLoader(list(data),
                          shuffle=shuffle,
                          batch_size=batch_size,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          )
    
    # -----------------------------------------------
    def predict(self,X):
        self.eval()
        with torch.no_grad():
            batch_size = min([1024,X.shape[0]])
            dataloader = self.data_loader(X,batch_size,shuffle=False)
            num_elements = X.shape[0]
            num_batches = len(dataloader)
            predictions = torch.zeros((num_elements,1),device=self.device)
            for i, batch in enumerate(dataloader):
                start = i*batch_size
                end = start + batch_size
                if i == num_batches - 1:
                    end = num_elements
                pred = self.model.forward(batch.to(self.device)).detach()
                predictions[start:end,:] = pred
        return predictions

    # -----------------------------------------------
    def fit(self,
            X_train,
            Y_train,
            X_val=None,
            Y_val=None,            
            epochs=1,
            batch_size=32,
            N_print=500,
            num_workers=0):

        # create the train_loeder object
        train_loader = self.data_loader(zip(X_train,Y_train),
                                        batch_size=batch_size,
                                        num_workers = num_workers)

        # Optimizer
        optimizer = optim.Adam(self.model.parameters())

        # if model.fit already called, do not reinitialize histories
        if hasattr(self, 'history_loss') == False:
            self.history_loss = []

        time_start = time.time()
        for epoch in range(epochs):  # loop over the dataset multiple times

            loss_epoch = []
            
            time_epoch_start = time.time()
            for i, data in enumerate(train_loader):
                X, y = data
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction and error
                pred = self.model.forward(X)   
                loss = nn.MSELoss()(pred, y)

                # Backpropagation
                optimizer.zero_grad()ß
                loss.backward()
                optimizer.step()

                # Gather data and report
                loss_epoch.append(loss.item())

                if i % N_print == (N_print-1): # print every N_print mini-batches
                    print(' epoch {} of {} - batch {} loss: {:5.4f} accuracy: {:5.4f}'.format(epoch+1, 
                                                                                              epochs,
                                                                                              i + 1,
                                                                                              np.mean(loss_epoch),
                                                                                            ))

            self.history_loss.append(np.mean(loss_epoch))  

            time_epoch_end = time.time() - time_epoch_start
            print('time per epoch: {:5.4f}'.format(time_epoch_end))

        print('Finished Training')
        self.time_training_end = time.time() - time_start

        # compute loss and accuracy for validation set
        if X_val is not None:
            # create the train_loeder object
            val_loader = self.data_loader(zip(X_val,Y_val),
                                      batch_size=min([1024,Y_val.shape[0]]),
                                      num_workers = num_workers)
            
            # Validation loss
            val_loss = []
            for i, data in enumerate(val_loader, 0):
                with torch.no_grad():
                    X, y = data
                    X, y = X.to(self.device), y.to(self.device)

                    pred = self.model(X)
                    val_loss.append(self.weighted_BCELoss(pred, y, self.class_weight))

            self.val_loss = np.mean(loss_epoch)


model = base_MLP(nun_hidden_layers=5,
                 in_features=2,
                 hidden_features=10)

model(X_train)

model_trainer = trainer(model)
model_trainer.model.visual_graph()



model_trainer.fit(X_train,
            Y_train,          
            epochs=500,
            batch_size=32,
            N_print=20)




X_test = torch.from_numpy(np.column_stack((xx.reshape(-1,1),yy.reshape(-1,1)))).float()
sol_ref = f_ref(X_test[:,0], X_test[:,1], f)

sol_PINN = model_trainer.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_test[:,0],X_test[:,1],sol_PINN,c = sol_PINN,marker = '*', alpha = 1)
# ax.scatter(X_test[:, 0], X_test[:, 1], sol_ref, marker='o', alpha=.1)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(sol_PINN, sol_ref.reshape(-1, 1))
print('mse =', mse)


plt.figure()
plt.contourf(xx, yy, np.reshape(
    abs(sol_PINN - sol_ref.reshape(-1, 1)), xx.shape), 30)
plt.axis('equal')
plt.colorbar()
plt.title('Absolute error')
plt.show()





















