from torch import nn
import base_MLP as MLP


class PINN_model(nn.Module, MLP):
    ''' Definition of PINN starting from basemodel MLP '''
    
    def __init__(self):
        super().__init__()


    def loss_PDE(self):
        raise NotImplementedError()

    def loss_BCSß(self):
        raise NotImplementedError()

    def loss(self,y_true):
        loss_PDE = self.loss_PDE(self)
        loss_BCS = self.loss_BCS(self)     
        return torch.mean(loss_PDE,loss_BCS)   