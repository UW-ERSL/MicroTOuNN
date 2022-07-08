import torch
#from torch_sparse_solve import solve
import numpy as np
import time
import matplotlib.pyplot as plt
import torch_sparse_solve
#------------------------------------------------------------------------#
def to_torch(x):
  return torch.tensor(x).double()
#------------------------------------------------------------------------#
def to_np(x):
  return x.detach().cpu().numpy()
#------------------------------------------------------------------------#
def getLameMaterialProperties(E, nu):
  lam, mu = np.zeros_like(E), np.zeros_like(E)
  for i in range(E.shape[0]):
    if (E[i] == 0):
        mu[i] = 0
        lam[i] = 0
    else:
        l1 = E[i]*nu[i]/((1+nu[i])*(1-2*nu[i]))
        m = E[i]/(2*(1+nu[i]))
        mu[i] = m
        lam[i] = 2*m*l1/(l1+2*m)
  return lam, mu
#------------------------------------------------------------------------# 
# def getLameMaterialProperties(E, nu):
#   l1 = E*nu/((1+nu)*(1-2*nu))
#   mu = E/(2*(1+nu))
#   lam = 2*mu*l1/(l1+2*mu)
#   return lam, mu
# #------------------------------------------------------------------------#
class Homogenize:
  def __init__(self, meshProp):
    self.meshProp = meshProp
    self.meshProp['numElems'] = meshProp['nelx']*meshProp['nely']
    self.meshProp['elemCenters'] = self.generatePoints(1)
    self.KFTemplates = \
      self.elementMatVec(0.5*meshProp['elemSize'][0],\
                         0.5*meshProp['elemSize'][1],\
                         meshProp['cellAngleDeg'])
    self.nodeIdx = self.computeStiffnessIndices()
    self.computeChi0()
    self.computeBCPenaltyMatrices()
    self.fig, self.ax = plt.subplots()
  #------------------------------------------------------------------------#
  def computeBCPenaltyMatrices(self):
    self.BCPenalty = {}
    # K matrix
    V = np.zeros((self.meshProp['ndof'], self.meshProp['ndof']))
    fixed = np.array([0,1]).astype(int)
    V[fixed,fixed] = 1.
    V = torch.tensor(V[np.newaxis])
    indices = torch.nonzero(V).t()
    values = V[indices[0], indices[1], indices[2]]
    
    self.BCPenalty['K'] = 1e12 *torch.sparse_coo_tensor(\
                                              indices, values, V.size())
    # F matrix
    V = np.zeros((self.meshProp['ndof'], 3))
    fixed = np.array([0,1]).astype(int)
    V[fixed,fixed] = 1.
    V = torch.tensor(V[np.newaxis])
    indices = torch.nonzero(V).t()
    values = V[indices[0], indices[1], indices[2]]
    
    self.BCPenalty['F'] = 1e12*torch.sparse_coo_tensor(\
                                              indices, values, V.size())
  #------------------------------------------------------------------------#
  def computeStiffnessIndices(self):
    nelx, nely = self.meshProp['nelx'], self.meshProp['nely']
    numElems = self.meshProp['numElems']
    nodenrs = np.reshape(np.arange(0,(1+nely)*(1+nelx)),(1+nelx,1+nely))
    edofVec = np.reshape(2*nodenrs[0:-1,0:-1]+2,(numElems,1))

    edofMat = np.repeat(edofVec,8 , axis = 1)
    edofMat = edofMat + np.repeat(np.array([0, 1, 2*nely+2, 2*nely+3,\
                                            2*nely, 2*nely+1, -2, -1])\
                                  [:,np.newaxis], numElems, axis = 1).T

    nn = (nelx+1)*(nely+1) # Total number of nodes
    nnP = (nelx)*(nely)   # Total number of unique nodes
    nnPArray = np.reshape(np.arange(0,nnP), (nelx, nely))
    nnPArray = np.vstack((nnPArray, nnPArray[0,:]))
    nnPArray = np.hstack((nnPArray, nnPArray[:,0][:,np.newaxis]))

    dofVector = np.zeros((2*nn))

    dofVector[0::2] = 2*nnPArray.flatten()
    dofVector[1::2] = 2*nnPArray.flatten()+1

    self.edofMat = dofVector[edofMat]
    self.meshProp['ndof'] = 2*nnP
    print("#FE DOF: ",self.meshProp['ndof'] );

    iK = np.kron(self.edofMat,np.ones((8,1))).T.flatten(order='F').astype(int)
    jK = np.kron(self.edofMat,np.ones((1,8))).T.flatten(order='F').astype(int)
    bK = tuple(np.zeros((len(iK))).astype(int)) #batch values
    KnodeIdx = np.array([bK,iK,jK])
    
    iF = np.tile(self.edofMat,3).T.flatten(order = 'F').astype(int)
    jF = np.tile(np.hstack((np.zeros(8),1*np.ones(8),2*np.ones(8))),numElems).astype(int)
    bF = tuple(np.zeros((len(iF))).astype(int)) #batch values
    FnodeIdx = np.array([bF,iF,jF])
    
    nodeIdx = {'K':KnodeIdx, 'F':FnodeIdx}
    return nodeIdx
  #------------------------------------------------------------------------#
  def generatePoints(self, res=1):
    # args: Mesh is dictionary containing nelx, nely, elemSize...
    # res is the number of points per elem
    # returns an array of size (numpts X 2)
    xy = np.zeros((res**2*self.meshProp['numElems'], 2))
    ctr = 0
    for i in range(res*self.meshProp['nelx']):
      for j in range(res*self.meshProp['nely']):
        xy[ctr, 0] = self.meshProp['elemSize'][0]*(i + 0.5)/(res)
        xy[ctr, 1] = self.meshProp['elemSize'][1]*(j + 0.5)/(res)
        ctr += 1
    return xy
    #--------------------------#
  def homogenize(self, netLam, netMu):
    nelx, nely, ndof = self.meshProp['nelx'], self.meshProp['nely'], self.meshProp['ndof']
    CH = torch.zeros((3,3))
    objElemAll = torch.zeros((nely,nelx,3,3))
    chi = self.computeDisplacements(netLam, netMu)

    for i in range(3):
      for j in range(3):
        vi = self.chi0[:,:,i] - chi[(self.edofMat+(i)*ndof)%ndof,\
                                    ((self.edofMat+(i)*ndof)//ndof)]
        vj = self.chi0[:,:,j] - chi[(self.edofMat+(j)*ndof)%ndof,\
                                    ((self.edofMat+(j)*ndof)//ndof)]
        sumLambda = torch.multiply(torch.mm(vi,self.KFTemplates['K']['lambda']),vj)
        sumMu = torch.multiply(torch.mm(vi,self.KFTemplates['K']['mu']),vj)

        sumLambda = torch.sum(sumLambda,1).view((nelx, nely)).T
        sumMu = torch.sum(sumMu,1).view((nelx, nely)).T

        objElemAll[:,:,i,j] = 1/self.meshProp['area']*(torch.multiply(netLam,sumLambda)\
                                                   + torch.multiply(netMu,sumMu))

        CH[i,j] = torch.sum(torch.sum(objElemAll[:,:,i,j]))

    return CH
  #------------------------------------------------------------------------#
  def computeDisplacements(self, lam, mu):

    tp1 = self.KFTemplates['K']['lambda'].T.flatten()
    tp2 = lam.T.flatten()
    tp3 = self.KFTemplates['K']['mu'].T.flatten()
    tp4 = mu.T.flatten()
    tp5 = (torch.outer(tp1,tp2) + torch.outer(tp3,tp4)).T.flatten()
    sK = tp5.double()

    K = torch.sparse_coo_tensor(self.nodeIdx['K'], sK.flatten(), \
            (1, self.meshProp['ndof'], self.meshProp['ndof']))
    
    tp1 = self.KFTemplates['F']['lambda'].T.flatten()
    tp3 = self.KFTemplates['F']['mu'].T.flatten()

    tp5 = (torch.outer(tp1,tp2) + torch.outer(tp3,tp4)).T.flatten()
    sF = tp5.double()

    F = torch.sparse_coo_tensor(self.nodeIdx['F'], sF.flatten(), \
            (1, self.meshProp['ndof'], 3))
    F_bc =  (F + self.BCPenalty['F']).coalesce().to_dense()

    dense = False
    if (dense):
        K_bc =  (K + self.BCPenalty['K']).coalesce().to_dense();
        chi = torch.linalg.solve(K_bc, F_bc)
    else:
        K_bc =  (K + self.BCPenalty['K']).coalesce();
        chi = torch_sparse_solve.solve(K_bc, F_bc)

    return chi[0,:,:]

  #------------------------------------------------------------------------#
  def plotMicrostructure(self, mcrstr, titleStr):
    plt.ion(); plt.clf();
    a = plt.imshow(mcrstr.reshape((self.meshProp['nelx'], self.meshProp['nely'])),\
                   cmap = 'gray')
    self.fig.canvas.draw()
    plt.colorbar(a)
    plt.title(titleStr)
    plt.pause(0.01)
  #------------------------------------------------------------------------#
  def computeChi0(self):
    
    self.chi0 = np.zeros((self.meshProp['numElems'], 8, 3))
    chi0_e = np.zeros((8, 3))

    ke = self.KFTemplates['K']['mu'] + self.KFTemplates['K']['lambda']
    fe = self.KFTemplates['F']['mu'] + self.KFTemplates['F']['lambda']

    idx = np.array([2,4,5,6,7])
    chi0_e[idx,:] = np.linalg.solve(ke[np.ix_(idx,idx)], fe[idx,:])
    
    for i in range(3):
      self.chi0[:,:,i] = np.kron(chi0_e[:,i], np.ones((self.meshProp['numElems'],1)))
    self.chi0 = to_torch(self.chi0)
  #------------------------------------------------------------------------#
  def elementMatVec(self, a, b, phi):
    CMu = np.diag([2, 2, 1])
    CLambda = np.zeros((3,3))
    CLambda[0:2,0:2] = 1
    
    xx = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gaussPoints = {'xx':xx, 'yy':xx, 'wt': np.array([1,1])}

    keLambda = np.zeros((8,8))
    keMu = np.zeros((8,8))
    feLambda = np.zeros((8,3))
    feMu = np.zeros((8,3))
    L = np.zeros((3,4))
    L[0,0] = 1
    L[1,3] = 1
    L[2,1:3] = 1
    for ii in range(gaussPoints['xx'].shape[0]):
      for jj in range(gaussPoints['yy'].shape[0]):
        x = gaussPoints['xx'][ii]
        y = gaussPoints['yy'][jj]
        dNx = 0.25*np.array([-(1-y), (1-y), (1+y), -(1+y)])
        dNy = 0.25*np.array([-(1-x), -(1+x), (1+x), (1-x)])
        Nvec = np.hstack((dNx,dNy)).T.reshape((2,4))
        Mtr = np.array([-a, -b, a, -b, a+2*b/np.tan(phi*np.pi/180), b, \
                        2*b/np.tan(phi*np.pi/180)-a, b]).reshape((4,2))
        J = np.einsum('ij,jk->ik', Nvec, Mtr)
        detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        invJ = 1/detJ*np.array([J[1,1], -J[0,1], -J[1,0], J[0,0]]).reshape((2,2))
        weight = gaussPoints['wt'][ii]*gaussPoints['wt'][jj]*detJ
        G = np.zeros((4,4))
        G[0:2,0:2] = invJ
        G[2:4, 2:4] = invJ
        dN = np.zeros((4,8))
        dN[0,0:8:2] = dNx
        dN[1,0:8:2] = dNy
        dN[2,1:8:2] = dNx
        dN[3,1:8:2] = dNy
        G_dN = np.einsum('ij,jk->ik', G, dN)
        B = np.einsum('ij,jk->ik', L, G_dN)
        keLambda = keLambda + weight*(np.dot(B.T, np.dot(CLambda, B))) # {8 X 8}
        keMu = keMu + weight*(np.dot(B.T, np.dot(CMu, B))) # {8 X 8}
        feLambda = feLambda + weight*(np.dot(B.T, np.dot(CLambda, np.diag([1, 1, 1])))) # {8 X 3}
        feMu = feMu + weight*(np.dot(B.T, np.dot(CMu, np.diag([1, 1, 1])))) # {8 X 3}

    return {'K':{'lambda': to_torch(keLambda), 'mu':to_torch(keMu)},\
            'F':{'lambda': to_torch(feLambda), 'mu':to_torch(feMu)}}

#%% Mesh
def test():
  nelx, nely = 100, 100
  lx, ly = 1., 1.
  meshProp = {'domSize':np.array([lx, ly]), 'cellAngleDeg': 90,\
              'elemSize':np.array([lx/nelx, ly/nely]), 'nelx':nelx, 'nely':nely,\
              'area': lx*ly}
  
  #%% Material
  E, nu = 1., 0.3 # Hooke material properties
  lam, mu = getLameMaterialProperties(E, nu)
  matProp = {'type':'lame', 'lam':lam, 'mu':mu, 'penal':3.}
  
  #%% 
  def square(nelx, nely, t):
    '''
    t between 0 and 1
    '''
    microStr = np.zeros((nelx,nely))
    t = 0.5*t*nelx # scale t
  
    for rw in range(nelx):
      for col in range(nely):
        if((rw < t) or (nelx-rw < t) or (col < t) or (nely-col < t)):
          microStr[rw,col] = 1
        else:
          microStr[rw,col] = 1e-4
    return microStr
  
  
  x = to_torch(square(nelx, nely, 0.15))
  
  H = Homogenize(meshProp, matProp)
  start = time.perf_counter()
  ch = H.homogenize(x)
  print('tiome ', time.perf_counter() - start)
  print(ch)
#test()