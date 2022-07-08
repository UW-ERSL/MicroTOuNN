import numpy as np
import random
import torch
import torch.nn as nn

def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
#%% Neural network
class TopNet(nn.Module):
    def __init__(self, nnSettings, inputDim):
        self.inputDim = inputDim; # x and y coordn of the point
        self.outputDim = nnSettings['outputDim']; # if material/void at the point
        super().__init__();
        self.layers = nn.ModuleList();
        manualSeed = 1234; # NN are seeded manually
        set_seed(manualSeed);
        current_dim = self.inputDim;
        for lyr in range(nnSettings['numLayers']): # define the layers
            l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr']);
            nn.init.xavier_normal_(l.weight);
            nn.init.zeros_(l.bias);
            self.layers.append(l);
            current_dim = nnSettings['numNeuronsPerLyr'];
        self.layers.append(nn.Linear(current_dim, self.outputDim));
        self.bnLayer = nn.ModuleList();
        for lyr in range(nnSettings['numLayers']): # batch norm
            self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']));

    def forward(self, x):
        m = nn.SiLU()
        ctr = 0;
        for layer in self.layers[:-1]: # forward prop
            x = m(self.bnLayer[ctr](layer(x)));
            ctr += 1;
        rho = 0.01 +torch.softmax(self.layers[-1](x), dim = 1)#.view(-1)
        return  rho
    def  getWeights(self): # stats about the NN
        modelWeights = [];
        modelBiases = [];
        for lyr in self.layers:
            modelWeights.extend(lyr.weight.data.view(-1).cpu().numpy());
            modelBiases.extend(lyr.bias.data.view(-1).cpu().numpy());
        return modelWeights, modelBiases;
