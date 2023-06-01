import torch
from torch import nn
import base_MLP as MLP


class HDL_Eni(nn.Module):
    def _init_(self,layer_list,device=None):
        super()._init_()
        # instantiate layers
        self.linears = nn.ModuleList(layer_list)

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device);

    # -----------------------------------------------
    def forward(self,x):
        x = x.to(self.device)
        for i,layer in enumerate(self.linears):
            x = layer(x)
        return x 
    
    # -----------------------------------------------
    def summary(self):
        return torchinfo.summary(self, input_size=(1,n_features,l_window),device=self.device)
        
    
    # -----------------------------------------------
    def visual_graph(self):
        model_graph = draw_graph(self, 
                                 input_size=(1,n_features,l_window), 
                                 expand_nested=True, 
                                 device=self.device)
        return model_graph.visual_graph
    
    @staticmethod
    def compute_classes_weight(Y):
        class_weight=compute_class_weight(class_weight = 'balanced',
                                          classes = np.unique(Y.numpy()),
                                          y = Y.numpy().ravel())
        class_weight = dict(zip(np.unique(Y.numpy().ravel()),class_weight))
        return class_weight
    
    # -----------------------------------------------
    def data_loader(self, data, batch_size, num_workers=0, pin_memory=True, shuffle=True):
        return DataLoader(list(data),
                          shuffle=shuffle,
                          batch_size=batch_size,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          )
    
    # -----------------------------------------------
    @staticmethod
    def weighted_BCELoss(pred,y,class_weight):
        eps=1e-7
        loss = class_weight[1]y*torch.log(pred+eps) + class_weight[0](1-y)*torch.log(1-pred+eps)
        return torch.neg(torch.mean(loss))
    
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
                pred = self.forward(batch.to(self.device)).detach()
                predictions[start:end,:] = pred
        return predictions
    
    # -----------------------------------------------
    def evaluate(self,X,Y,metrics):
        pass

    # -----------------------------------------------
    @staticmethod
    def get_windowed_data_shot(X_shot):
        X_shot_windowed = X_shot.unfold(dimension=-1, size=l_window, step=1)
        return torch.transpose(X_shot_windowed, dim0=1, dim1=0)
    # -----------------------------------------------
    def predict_shot(self,X_shot):
        return self.predict(self.get_windowed_data_shot(X_shot)).cpu().numpy()
        
    # -----------------------------------------------
    def evaluate_shot(self,X_shot,Y_shot,metrics):
        pass

    # -----------------------------------------------
    def fit(self,
            X_train,
            Y_train,
            X_val=None,
            Y_val=None,            
            epochs=1,
            batch_size=32,
            class_weight = None,
            N_print=500,
            num_workers=0):

        # define class_weight for unbalance problems
        if class_weight == None:
            self.class_weight = self.compute_classes_weight(Y_train)
        else:
            self.class_weight = dict({0:1,1:1})

        # create the train_loeder object
        train_loader = self.data_loader(zip(X_train,Y_train),
                                      batch_size=batch_size,
                                      num_workers = num_workers)

        # Optimizer
        optimizer = optim.Adam(self.parameters())

        # if model.fit already called, do not reinitialize histories
        if hasattr(self, 'history_loss') == False:
            self.history_loss = []
            self.history_accuracy = []

        time_start = time.time()
        for epoch in range(epochs):  # loop over the dataset multiple times

            loss_epoch = []
            accuracy_epoch = []
            
            time_epoch_start = time.time()
            for i, data in enumerate(train_loader):
                X, y = data
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction and error
                pred = self.forward(X)   
                loss = self.weighted_BCELoss(pred, y, self.class_weight)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Gather data and report
                loss_epoch.append(loss.item())

                # compute accuracy
                accuracy_epoch.append(100*(torch.round(pred.detach().data) == y).sum().item()/y.shape[0])

                if i % N_print == (N_print-1): # print every N_print mini-batches
                    print(' epoch {} of {} - batch {} loss: {:5.4f} accuracy: {:5.4f}'.format(epoch+1, 
                                                                                         epochs,
                                                                                         i + 1,
                                                                                         np.mean(loss_epoch),
                                                                                         np.mean(accuracy_epoch)
                                                                                         ))

            self.history_loss.append(np.mean(loss_epoch))  
            self.history_accuracy.append(np.mean(accuracy_epoch))  

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
            val_accuracy = []
            val_loss = []
            for i, data in enumerate(val_loader, 0):
                with torch.no_grad():
                    X, y = data
                    X, y = X.to(self.device), y.to(self.device)

                    pred = self(X)
                    val_loss.append(self.weighted_BCELoss(pred, y, self.class_weight))
                    val_accuracy.append(100*(torch.round(pred.data) == y).sum().item()/y.shape[0])

            self.val_loss = np.mean(loss_epoch)
            self.val_accuracy = np.mean(val_accuracy)