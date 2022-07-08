import numpy as np
import torch
from torchHomogenization import getLameMaterialProperties, Homogenize, to_torch,\
  to_np
import matplotlib.pyplot as plt
import time

#--------------------------#
#%% Filter
def computeFilter(nx, ny, rmin):
  H = np.zeros((nx*ny,nx*ny));

  for i1 in range(nx):
    for j1 in range(ny):
      e1 = (i1)*ny+j1;
      imin = max(i1-(np.ceil(rmin)-1),0.);
      imax = min(i1+(np.ceil(rmin)),nx);
      for i2 in range(int(imin), int(imax)):
        jmin = max(j1-(np.ceil(rmin)-1),0.);
        jmax = min(j1+(np.ceil(rmin)),ny);
        for j2 in range(int(jmin), int(jmax)):
          e2 = i2*ny+j2;
          H[e1, e2] = max(0.,rmin-\
                             np.sqrt((i1-i2)**2+(j1-j2)**2));

  Hs = np.sum(H,1);
  return H, Hs;
#--------------------------#
def applySensitivityFilter(ft, x, dc, dv):
  if (ft['type'] == 1):
    dc = np.matmul(ft['H'],\
                     np.multiply(x, dc)/ft['Hs']/np.maximum(1e-3,x));
  elif (ft['type'] == 2):
    dc = np.matmul(ft['H'], (dc/ft['Hs']));
    dv = np.matmul(ft['H'], (dv/ft['Hs']));
  return dc, dv;

#%% Boundary condition
#--------------------------#
#--------------------------#
def oc(rho, dc, dv, ft, vf):
  l1 = 0; 
  l2 = 1e9;
  x = rho.copy();
  move = 0.2;
  while (l2-l1 > 1e-4):
    lmid = 0.5*(l2+l1);
    dr = np.abs(-dc/dv/lmid);

    xnew = np.maximum(0,np.maximum(x-move,\
                    np.minimum(1,np.minimum(\
                    x+move,x*np.sqrt(dr)))));
    if ft['type'] == 1:
      rho = xnew;
    elif ft['type'] == 2:
      rho = np.matmul(ft['H'],xnew)/ft['Hs'];
    if np.mean(rho) > vf:
      l1 = lmid; 
    else:
      l2 = lmid; 

  change =np.max(np.abs(xnew-x));
  return rho, change;
#--------------------------#  

def optimize(H, vf, ft, maxIter = 200):
  nelx, nely = H.meshProp['nelx'], H.meshProp['nely']
  rho = vf*torch.ones((H.meshProp['numElems']), requires_grad = True)
  change, loop = 10., 0
  t0 = time.perf_counter()
  while(loop < maxIter): # change > 0.0001 and
    rho.retain_grad()
    loop += 1
    
    ch = H.homogenize(rho.view((nelx, nely)))
    c = H.computeObjective(ch)
    c.backward()
    dc = to_np(rho.grad)
    rho = to_np(rho)
    dv = np.ones((nelx*nely))
    dc, dv = applySensitivityFilter(ft, rho, dc, dv)

    rho, change = oc(rho, dc, dv, ft, vf)
    
    status = 'iter {:d} ;  obj {:.2F} ; vol {:.2F}'.format(\
            loop,  c, np.mean(rho))
    if(loop % 20 == 0):
      plt.imshow(-rho.reshape((nelx,nely)),\
                 cmap = 'gray')
      plt.title(status)
      plt.show()
      plt.pause(0.01)
    rho = torch.tensor(rho, requires_grad = True)

    print(status, 'change {:.2E}'.format(change))
  print('time taken (sec): ', time.perf_counter() - t0)

    
def test():
  nelx, nely = 40, 40
  lx, ly = 40., 40.
  meshProp = {'domSize':np.array([lx, ly]), 'cellAngleDeg': 90,\
              'elemSize':np.array([lx/nelx, ly/nely]), 'nelx':nelx, 'nely':nely,\
              'area': lx*ly}
  
  #%% Material
  E, nu = 1., 0.3 # Hooke material properties
  lam, mu = getLameMaterialProperties(E, nu)
  matProp = {'type':'lame', 'lam':lam, 'mu':mu, 'penal':3.}
  
  Fltr, FltrSum = computeFilter(nelx, nely, rmin = 5.)
  ft = {'type':1, 'H':Fltr, 'Hs':FltrSum}
  
  vf = 0.5
  H = Homogenize(meshProp, matProp)
  optimize(H, vf, ft, maxIter = 200)
  
test()
  
  
  