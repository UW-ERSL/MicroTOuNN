import numpy as np

from microTOuNN import TopologyOptimizer
import matplotlib.pyplot as plt
from torchHomogenization import getLameMaterialProperties

# Factors that effect the final design, with defaults
# 1. Mesh size  (60 x 60)
# 2. Neural network size (5 x 30)
# 3. Activative function (SiLU)
# 4. Alpha parameters in AugLag (1 + 5)
# 5. Start image (squareSolid)
# 6. SIMP Penalty with continuation (2 + 0.5)

def HashinShtrikman(E0,nu,volFrac):
    #Topology optimization of micro-structured materials featured with ...
    # Note volFrac is refers to the base material 
        K0 = E0/(3*(1-2*nu));
        G0 = E0/(2*(1+nu));
        KStar = volFrac*K0*G0/((1-volFrac)*K0 + G0)
        GStar =  G0 + (1-volFrac)/(-1/G0 +6*(K0+2*G0)*volFrac/(5*G0*(3*K0+4*G0)))
        return [KStar, GStar] 
    
def checkConstraints(objective,constraints):
    if (objective == 'K') and (constraints['KConstraint'][1]):
        print("With K-objective, KConstraint should be set to False ")
        return False
    if (objective == 'G') and (constraints['GConstraint'][1]):
        print("With G-objective, GConstraint should be set to False ")
        return False
    if (objective == 'nu') and (constraints['nuConstraint'][1]):
        print("With nu-objective, nuConstraint should be set to False ")
        return False
    if (objective == 'm') and (constraints['massConstraint'][1]):
        print("With m-objective, mConstraint should be set to False ")
        return False
    if (not constraints['KConstraint'][1]) and \
        (not constraints['GConstraint'][1]) and \
        (not constraints['nuConstraint'][1]) and \
        (not constraints['CConstraint'][1]) and \
        (not constraints['massConstraint'][1]) :
        print("At least one constraint should to True ")
        return False
    
    print("Objective and constraints are OK ")
    return True
#%% Mesh
nelx, nely = 60,60
lx, ly = 1., 1.
meshProp = {'domSize':np.array([lx, ly]), 'cellAngleDeg': 90,\
            'elemSize':np.array([lx/nelx, ly/nely]), 'nelx':nelx, 'nely':nely,\
            'area': lx*ly}

#%% Material
nMaterials = 1 # the number of non-void materials,
EVoid = 1e-3 # Young's modulus of void material, non-zero value required
rhoVoid = 0 #  density of void material, zero value OK
PoissonVoid = 0.3
if (nMaterials == 1):
    E, nu = np.array([EVoid,1]), np.array([PoissonVoid, 0.3]) # +1 for void
    massDens = np.array([rhoVoid, 1])# +1 for void
elif (nMaterials == 2):
    E, nu = np.array([EVoid,1., 0.2]), np.array([PoissonVoid, 0.3, 0.3]) #  +1 for void
    massDens = np.array([rhoVoid, 1., 0.2 ])# +1 for void
elif (nMaterials == 3):
     E, nu = np.array([EVoid, 1.,0.2, 0.3]), np.array([PoissonVoid, 0.3, 0.3, 0.3]) #  +1 for void
     massDens = np.array([rhoVoid, 1., 0.2, 0.25]) # +1 for void  
elif (nMaterials == 4):
     E, nu = np.array([EVoid, 1.,0.2, 0.3,0.4]), np.array([PoissonVoid,0.3, 0.3, 0.3, 0.3]) #  +1 for void
     massDens = np.array([rhoVoid, 1., 0.2,0.25, 0.3]) # +1 for void  
        
lam, mu = getLameMaterialProperties(E, nu)
matProp = {'type':'lame', 'lam':lam, 'mu':mu, 'massDens':massDens, 'penal':2,\
           'penalincrement':0.5,'penalmax':10} # SIMP continuation
nnSettings = {'numLayers': 5, 'numNeuronsPerLyr': 30, 'outputDim': E.size }
#%% Optimization Problems

objective =  'K' ## K (maximize), G (maximize), or nu (minimize) or m (minimize) or '' (no objective)

# Not all of these constraints may be active, see below for True or False
# value and whether it is imposed
constraints= { "KConstraint": [0.09,False], 
            'GConstraint': [0.01,False],\
            'nuConstraint': [0.3,False],\
            "CConstraint": [[[0.13,-0.03,0],[-0.03, 0.13, 0],[0,0,0.015]],False],\
            'massConstraint': [0.37*nelx*nely*max(massDens), True],\
            '45DegreeSymmetry':True}

startImageType = 'circularSolid' ## none squareSolid squareHole circularSolid circularHole 4circularHoles
augLagParams = {"alpha0":1, "alphaIncrement":5}
if (checkConstraints(objective,constraints)):
    minEpochs = 3 # minimum number of iterations
    maxEpochs = 50 # Max number of iterations
    plt.close('all')
    useCPU = True
    if (nMaterials == 1):
        volFrac = constraints['massConstraint'][0]/(nelx*nely*max(massDens))
        [KStar,GStar] = HashinShtrikman(1,0.3,volFrac)
        print('----------------')
        print("volFrac: {:.2F}".format( volFrac))
        print("KStar: {:.4F}".format( KStar))
        print("GStar: {:.4F}".format( GStar))
        print('----------------')
    plt.figure();
    topOpt = TopologyOptimizer(meshProp, matProp, nnSettings,\
                                objective,constraints,startImageType,augLagParams,\
                                useCPU)
    topOpt.optimizeDesign(maxEpochs,minEpochs)
    plt.plot(topOpt.objectiveHistory['K'], 'b',label = 'K')
    plt.plot(np.array(topOpt.objectiveHistory['G']), 'g:',label = 'G')
    plt.plot(np.array(topOpt.objectiveHistory['nu']), 'r--',label = 'nu')
    plt.plot(np.array(topOpt.objectiveHistory['mass']), 'k-.',label = 'mass')
    plt.xlabel('FE Operations');
    plt.grid('True')
    plt.legend(loc='upper left', shadow=True, fontsize='large')
    
    plt.figure();
    plt.plot(topOpt.errorHistory['objError'], 'b',label = 'Objective error')
    plt.plot(topOpt.errorHistory['constraintError'], 'r:',label = 'Constraint error')

    plt.xlabel('BFGS Iterations');
    plt.grid('True')
    plt.legend(loc='upper right', shadow=True, fontsize='large')

 
