#Versions
#Numpy 1.18.1
#Pytorch 1.5.0
#scipy 1.4.1
#cvxopt 1.2.0

#%% imports
import time
import numpy as np
import torch
import torch.optim as optim
from torchHomogenization import Homogenize
from plotUtil import Plotter
from network import TopNet
#from torch.autograd import grad
#import time
import matplotlib.pyplot as plt
from matplotlib import colors

class TopologyOptimizer:
    #-----------------------------#

  def __init__(self,  meshProp, matProp, nnSettings,\
               desiredObjective,constraints,startImageType, augLagParams,\
                   useCPU = True):

    self.device = self.setDevice(useCPU)
    self.boundaryResolution  = 3 # default value for plotting and interpreting
    self.matProp = matProp
    self.H = Homogenize(meshProp)
    self.xy = torch.tensor(self.H.meshProp['elemCenters'], requires_grad = True).\
                                    float().view(-1,2).to(self.device)
    self.xyPlot = torch.tensor(self.H.generatePoints(self.boundaryResolution),\
                    requires_grad = True).float().view(-1,2).to(self.device)
    self.Pltr = Plotter()

    self.desiredObjective = desiredObjective
    self.desiredK = constraints['KConstraint'][0]
    self.isKConstraintImposed = constraints['KConstraint'][1]
    self.desiredG = constraints['GConstraint'][0]
    self.isGConstraintImposed = constraints['GConstraint'][1]
    self.desiredNu = constraints['nuConstraint'][0]
    self.isNuConstraintImposed = constraints['nuConstraint'][1]   
    self.desiredMass = constraints['massConstraint'][0]
    self.isMassConstraintImposed = constraints['massConstraint'][1]
    self.desiredC = torch.from_numpy(np.array(constraints['CConstraint'][0]))
    self.isCConstraintImposed = constraints['CConstraint'][1]
    
    self.is45DegreeSymmetryImposed = constraints['45DegreeSymmetry']
    
    self.startImageType = startImageType

    self.nMaterials = self.matProp['massDens'].shape[0]-1 # not counting void
    self.nelx = meshProp['nelx']
    self.nely = meshProp['nely']
    self.elemSize =   meshProp['elemSize']
    print("#Elements: ",self.nelx * self.nely);
    print("#Materials: ",self.nMaterials);
    inputDim = 2
    self.topNet = TopNet(nnSettings, inputDim).to(self.device)
    modelWeights, modelBiases = self.topNet.getWeights();
    print("#Design variables: ",len(modelWeights) + len(modelBiases));
    self.objective = 0.
    self.alpha0 = augLagParams['alpha0']
    self.alphaIncrement = augLagParams['alphaIncrement']

    self.fig, self.ax = plt.subplots()
    #self.rotationalSymmetry = {'isOn' : True, 'centerCoordn': np.array([0.5*self.nelx*self.elemSize[0], 0.5*self.nely*self.elemSize[1]]), 'sectorAngleDeg':90}
  #-----------------------------#
  def setDevice(self, useCPU):
    if(torch.cuda.is_available() and (useCPU == False) ):
      device = torch.device("cuda:0")
      print("Running on GPU")
    else:
      device = torch.device("cpu")
      print("Running on CPU")
    return device
  #-----------------------------#
  def applyRotationalSymmetry(self, xyCoordn,degrees):  
      dx = xyCoordn[:,0] - 0.5*self.nelx*self.elemSize[0]
      dy = xyCoordn[:,1] - 0.5*self.nelx*self.elemSize[1]
      radius = torch.sqrt((dx)**2 + (dy)**2)
      angle = torch.atan2(dy, dx)
      correctedAngle = torch.remainder(angle, np.pi*degrees/180.)
      x, y = radius*torch.cos(correctedAngle), radius*torch.sin(correctedAngle)
      xyCoordn = torch.transpose(torch.stack((x,y)),0,1)
      return xyCoordn
  #-----------------------------#
  def applyXYSymmetry(self, xyCoordn):
      xv = 0.5*self.nelx*self.elemSize[0] + torch.abs(xyCoordn[:,0] - 0.5*self.nelx*self.elemSize[0]);
      yv = 0.5*self.nely*self.elemSize[1] + torch.abs(xyCoordn[:,1] - 0.5*self.nely*self.elemSize[1]) ;
      xyCoordn = torch.transpose(torch.stack((xv, yv)), 0, 1)
      return xyCoordn
  #-----------------------------#
  def apply45DegreeSymmetry(self, xyCoordn):
      x = xyCoordn[:,0] - 0.5*self.nelx*self.elemSize[0]
      y = xyCoordn[:,1] - 0.5*self.nelx*self.elemSize[1]
      delta = y-x
      xN = x + 0.5*(torch.abs(delta)+ delta) # swap x and y if y > x
      yN = y - 0.5*(torch.abs(delta)+ delta)
      xv = 0.5*self.nelx*self.elemSize[0] + xN;
      yv = 0.5*self.nely*self.elemSize[1] + yN;
      xyCoordn = torch.transpose(torch.stack((xv, yv)), 0, 1)
      return xyCoordn
  #-----------------------------#
  def optimizeDesign(self,maxEpochs, minEpochs):
    def getStartImage(relSize=0.2):
        xc, yc = 0.5*self.nelx*self.elemSize[0], 0.5*self.nely*self.elemSize[1]
        startImg = 0.5*np.ones((self.xy.shape[0]))
        if (self.startImageType == 'uniformGrey'):
            startImg =relSize*np.ones((self.xy.shape[0]))
        elif (self.startImageType== 'squareHole'):
            r = relSize*self.nelx*self.elemSize[0]
            startImg =np.zeros((self.xy.shape[0]))
            for i in range(startImg.shape[0]):
                  if((abs(self.xy[i,0] - xc)<r) and (abs(self.xy[i,1] - yc)<r)):
                      startImg[i] = 1
        elif (self.startImageType== 'squareSolid'):
            r = relSize*self.nelx*self.elemSize[0]
            startImg =np.ones((self.xy.shape[0]))
            for i in range(startImg.shape[0]):
                  if((abs(self.xy[i,0] - xc)<r) and (abs(self.xy[i,1] - yc)<r)):
                      startImg[i] = 0
        elif (self.startImageType == 'circularHole'):
           r = relSize*self.nelx*self.elemSize[0]
           startImg = np.zeros((self.xy.shape[0]))
           for i in range(startImg.shape[0]):
               if((self.xy[i,0] - xc)**2 + (self.xy[i,1] - yc)**2 <= r**2):
                   startImg[i] = 1
        elif (self.startImageType == 'circularSolid'):
           r = relSize*self.nelx*self.elemSize[0]
           startImg = np.ones((self.xy.shape[0]))
           for i in range(startImg.shape[0]):
               if((self.xy[i,0] - xc)**2 + (self.xy[i,1] - yc)**2 <= r**2):
                   startImg[i] = 0
        elif (self.startImageType == '4circularHoles'):
           r = 0.25*relSize*self.nelx*self.elemSize[0]
           startImg = np.zeros((self.xy.shape[0]))
           for i in range(startImg.shape[0]):
            if((self.xy[i,0] - xc/2)**2 + (self.xy[i,1] - yc/2)**2 <= r**2):
                   startImg[i] = 1
            if((self.xy[i,0] - 3*xc/2)**2 + (self.xy[i,1] - yc/2)**2 <= r**2):
                startImg[i] = 1
            if((self.xy[i,0] - xc/2)**2 + (self.xy[i,1] - 3*yc/2)**2 <= r**2):
                startImg[i] = 1
            if((self.xy[i,0] - 3*xc/2)**2 + (self.xy[i,1] - 3*yc/2)**2 <= r**2):
                startImg[i] = 1        
        return startImg
    def computeC(nu):
        #%C = [1 nu 0; nu 1 0;0 0 (1-nu)/2] % check
        C = np.zeros((3,3))
        C[0,0] = 1
        C[0,1] = nu
        C[1,0] = nu
        C[1,1] = 1
        C[2,2] = (1-nu)/2
        return C
    
    def extractElasticityConstants(C):
        #%C = E0/(1-nu^2)*[1 nu 0; nu 1 0;0 0 (1-nu)/2] % check
        nu = (C[0,1]+C[1,0])/(C[0,0]+C[1,1])
        E = (C[0,0]+C[1,1])*(1-nu*nu)/2
        G = C[2,2]
        K = (C[0,0]+C[1,1]+C[0,1] + C[1,0])/4
        return [E,nu,K,G]
    
    def computeEffectiveMaterialProperty(x):
      netLam = torch.zeros((x.shape[0])) # vec of size numelements
      netMu = torch.zeros((x.shape[0]))
      for i in range(self.matProp['lam'].shape[0]): # for every material...
        netLam = netLam + self.matProp['lam'][i]*(x[:,i]**self.matProp['penal']) # add their contribution
        netMu = netMu + self.matProp['mu'][i]*(x[:,i]**self.matProp['penal'])    
      return netLam, netMu

    def computeElementMasses(x):
      elemMass= torch.zeros((x.shape[0]))
      for i in range(self.matProp['massDens'].shape[0]):
          elemMass = elemMass + self.matProp['massDens'][i]*x[:,i]
      return elemMass
    
    def initializeNNWeights():
      if (self.startImageType != 'none'):
          startImg = torch.tensor(getStartImage()).float()
          # for this, simple Adam optimizer is sufficient
          self.optimizer = optim.Adam(self.topNet.parameters(), amsgrad=True,lr=0.01);  
          for epoch in range(250):
                self.optimizer.zero_grad()
                nn_rho = self.topNet(self.xy).to(self.device).double()
                if (self.nMaterials == 1):
                    nn_rho = nn_rho[:,0]
                else:
                    nn_rho = torch.mean(nn_rho,axis=1)
                loss = torch.mean((nn_rho-startImg)**2)
                loss.backward()
                self.optimizer.step()
          self.H.plotMicrostructure(nn_rho.detach().cpu(), ' ');
      return 
    ## Main optimization starts here
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True) 
    # first initialize NN weights so that the image is as desired
    initializeNNWeights()
    # Use L-BFGS for the main optimization
    self.optimizer = optim.LBFGS(self.topNet.parameters(),\
                                 line_search_fn = 'strong_wolfe')
    
    if (self.isKConstraintImposed):
        alphaK = self.alpha0  # for
        muK = 0; # lagrange multiplier
    if (self.isGConstraintImposed):
        alphaG = self.alpha0  
        muG = 0; 
    if (self.isNuConstraintImposed):
        alphaNu = self.alpha0  
        muNu = 0 # for nu constraint
    if (self.isMassConstraintImposed):
        alphaMassConstraint = self.alpha0  
        muMassConstraint = 0;
    if (self.isCConstraintImposed):
        alphaCConstraint = 10*self.alpha0  
        muCConstraint = 0;      
    xy = self.xy
    xy = self.applyXYSymmetry(xy) 
    if (self.is45DegreeSymmetryImposed):
        xy = self.apply45DegreeSymmetry(xy)  

    prevObj = 0
    prevConstraintError=0
    print(" ");
    self.objectiveHistory = {'K':[],'G':[], 'nu':[],'mass':[]}
    self.errorHistory  = {'objError':[],'constraintError':[]}    
    def closure():
          self.optimizer.zero_grad()       
          self.nn_rho = self.topNet(xy).to(self.device).double()
          netLam, netMu = computeEffectiveMaterialProperty(self.nn_rho)
          elemMass = computeElementMasses(self.nn_rho)
          self.mass = torch.sum(elemMass)
          ch = self.H.homogenize(netLam.view((self.H.meshProp['nelx'], self.H.meshProp['nely'])),\
                                 netMu.view((self.H.meshProp['nelx'], self.H.meshProp['nely'])))
          self.ch = ch     
          temp = extractElasticityConstants(self.ch)
          nu = temp[1]
          K = temp[2]
          G = temp[3]
          if (self.desiredObjective == 'K'):
               self.objective = -(ch[0,0]+ch[1,1]+ch[0,1]+ch[1,0])/4
          elif (self.desiredObjective == 'G'):
               self.objective = -ch[2,2]
          elif (self.desiredObjective == 'nu'):
               nu = (ch[0,1]+ch[1,0])/(ch[0,0]+ch[1,1])
               self.objective =  nu
          elif (self.desiredObjective == 'm'):
              self.objective = self.mass/(self.H.meshProp['nelx'] *self.H.meshProp['nely'])
          elif (self.desiredObjective == ''):
                  self.objective = 1e-5*torch.ones(1)
          loss = self.objective # accumulate different loss     
          self.currentMassFraction = self.mass.item()/(self.H.meshProp['nelx'] *self.H.meshProp['nely'])
          self.constraintError = 0# accumulate different constraint errors
          if (self.isKConstraintImposed):
            self.KConstraint =  K/self.desiredK-1
            self.constraintError = self.constraintError + abs(self.KConstraint.item())
            loss = loss + alphaK*torch.pow(self.KConstraint,2) + muK*self.KConstraint;     
          if (self.isGConstraintImposed):
            self.GConstraint =  G/self.desiredG-1
            self.constraintError = self.constraintError + abs(self.GConstraint.item())
            loss = loss + alphaG*torch.pow(self.GConstraint,2) + muG*self.GConstraint;            
          if (self.isNuConstraintImposed):
            self.nuConstraint =  nu/self.desiredNu-1
            self.constraintError = self.constraintError + abs(self.nuConstraint.item())
            loss = loss + alphaNu*torch.pow(self.nuConstraint,2) + muNu*self.nuConstraint;       
          if (self.isMassConstraintImposed):
            self.massConstraint =(self.mass/self.desiredMass.item()) - 1.0
            self.constraintError = self.constraintError + abs(self.massConstraint.item())
            loss = loss + alphaMassConstraint*torch.pow(self.massConstraint,2) + muMassConstraint*self.massConstraint;        
          if (self.isCConstraintImposed):
             self.CConstraint = 10*torch.norm(self.ch - self.desiredC)
             self.constraintError = self.constraintError + abs(self.CConstraint.item())
             loss = loss + alphaCConstraint*torch.pow(self.CConstraint,2) + muCConstraint*self.CConstraint;
          loss.backward(retain_graph=True);
          self.objectiveHistory['K'].append(K.item())
          self.objectiveHistory['G'].append(G.item())
          self.objectiveHistory['nu'].append(nu.item())
          self.objectiveHistory['mass'].append(self.currentMassFraction)
          return loss
    start = time.perf_counter()
    for epoch in range(maxEpochs):
          closure()  
          self.optimizer.step(closure)  
          if (self.isKConstraintImposed):
              muK = muK +  alphaK*2*self.KConstraint.item()
              alphaK = alphaK + self.alphaIncrement              
          if (self.isGConstraintImposed):
              muG = muG +  alphaG*2*self.GConstraint.item()
              alphaG = alphaG + self.alphaIncrement              
          if (self.isNuConstraintImposed):
              muNu = muNu +  alphaNu*2*self.nuConstraint.item()
              alphaNu = alphaNu + self.alphaIncrement         
          if (self.isMassConstraintImposed):
              muMassConstraint = muMassConstraint +  alphaMassConstraint*2*self.massConstraint.item()
              alphaMassConstraint = alphaMassConstraint + self.alphaIncrement          
          if (self.isCConstraintImposed):
              muCConstraint = muCConstraint +  alphaCConstraint*2*self.CConstraint.item()
              alphaCConstraint = alphaCConstraint + self.alphaIncrement
              
          self.matProp['penal'] = min(self.matProp['penalmax'],self.matProp['penal'] + self.matProp['penalincrement'] ); # continuation scheme 
          temp = extractElasticityConstants(self.ch)
          nu = temp[1].item()
          K = temp[2].item()
          G = temp[3].item()
          titleStr = "K:{:.3f},G:{:.3f},nu:{:.2f},m:{:.2f}".format(K,G,nu,self.currentMassFraction)
          self.plotMultimaterialDesign(titleStr, 1)
          # Error in objective
          objError = abs( (prevObj -self.objective.item())/self.objective.item())     
          self.errorHistory['objError'].append(objError)
          self.errorHistory['constraintError'].append(abs(self.constraintError))
          iterStr = "Iter: {:d}, ObjError: {:.3f}, constraintError: {:.3F}".format(epoch, \
                 abs(objError), abs(self.constraintError))
          print(iterStr);
          if (epoch >= minEpochs) and (objError <= 0.025) and (self.constraintError < 0.025):
              break
          if (epoch >= minEpochs) and (abs( (prevObj -self.objective.item()) <= 0.0001) and (abs( (prevConstraintError -self.constraintError))) <= 0.0001):
              print("Possible stagnation")
              break
          
          if (self.currentMassFraction > 0.99) or (self.currentMassFraction < 0.05):
               print("Possible divergence; try changing augLagrangian parameters and/or startImageType")
               break    
          prevObj = self.objective.item()
          prevConstraintError = self.constraintError
    print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))
    print(np.array(self.ch.tolist()));
    self.plotMultimaterialDesign(titleStr, 5)
    return self.objectiveHistory
  #-----------------------------#
  def plotMultimaterialDesign(self, titleStr, res = 1):
    if(res == 1):
      xy = self.xy
      rho = self.nn_rho.detach().cpu().numpy() 
    else: # dont gen pts unnecessarily
      xy = torch.tensor(self.H.generatePoints(res),\
                    requires_grad = True).float().view(-1,2).to(self.device)
      xyS = self.applyXYSymmetry(xy) 
      if (self.is45DegreeSymmetryImposed):
          xyS = self.apply45DegreeSymmetry(xyS)
      rho = self.topNet(xyS).to(self.device).double().detach().cpu().numpy()   
    plt.ion(); plt.clf();
    fillColors = ['white','black','red','green','blue'] 
    maxC = 0
    colorImg = np.zeros((res*self.H.meshProp['nelx'], res*self.H.meshProp['nely']))
    for elem in range(res**2*self.H.meshProp['numElems']):
      c = np.argmax(rho[elem,:])
      cx = int((res*xy[elem,0])/self.H.meshProp['elemSize'][0])
      cy = int((res*xy[elem,1])/self.H.meshProp['elemSize'][1])
      if(c > maxC):
        maxC = c
      colorImg[cx, cy] = c
    plt.imshow(colorImg.T, cmap = colors.ListedColormap(fillColors[:maxC+1]),\
               interpolation='none',vmin=0, vmax=maxC, origin = 'lower') 
          
    plt.title(titleStr,fontsize ='xx-large')
    plt.show()
    plt.pause(0.001)
    self.fig.canvas.draw()
    
  
  