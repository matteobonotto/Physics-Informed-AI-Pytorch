import torch
from torch import nn

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
        for i,layer in self.layers_hid:
            x = layer(x)
        return self.layer_out(x)


    # -----------------------------------------------
    def summary(self):
        return torchinfo.summary(self, input_size=self.input_size,device=self.device)
        
    
    # -----------------------------------------------
    def visual_graph(self):
        model_graph = draw_graph(self, 
                                 input_size=(1,n_features,l_window), 
                                 expand_nested=True, 
                                 device=self.device)
        return model_graph.visual_graph
    



