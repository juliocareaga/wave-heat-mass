#=====================================================================
# Code by Julio Careaga
# latest version: september 2024
# main code developed for the paper: Westervelt-based modeling of 
#                                    ultrasound-enhanced drug delivery
# co-authors: V. Nikolić and B. Said-Houari
#---------------------------------------------------------------------
from dolfinx      import plot, mesh, fem, io, geometry
from dolfinx.io   import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square, locate_entities, meshtags
import dolfinx.fem.petsc as dfpet
##--------------------------------------------------------------------
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import ufl
import math
import pyvista
import sys
import os
import scipy.io
try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    exit(0)

## The version (Wave-Eq)/(B(Theta)) has been commented; 
## search for the symbol if : (***)/B(Theta)

#---------------------------------------------!
# Time discretization: 
#    wave (linearized:fixed-point): Newmark;
#    heat: Implicit-Euler;
#    mass: Implicit-Euler;
# Space discretization:
#    wave: continuous Galerkin;
#    heat: continuous Galerkin;
#    mass: continuous Galerkin;
#---------------------------------------------!

## Parameters and functions:

DOLFIN_EPS = 3.0e-16

thetaa   = 37.0 # 37ºC or 310.15 K
betaacou = 6.0  # liver; Connor2002, table 3, variable beta
rho    = 1050.0 # taken as liver; 
rhoa   = 1050.0 # liver; Connor2002, table 3
rhob   = 1030.0 # blood; Connor2002, table 3
Ca     = 3600.0 # liver; Connor2002, table 3, variable C
Cb     = 3620.0 # blood; Connor2002, table 3, variable C
kappaa = 0.512  # liver; Connor2002, table 3, variable k
frec   = 100000        # frequency function g(t)
wfrec  = 2*np.pi*frec  # omega function g(t)
gg0    = 1.0e+9        # amplitude function g(t)
omega  = 2*np.pi*frec  # angular frequency;
alpha  = (4.5/1e6)*frec # liver; Connor2002, table 3; (alpha_0/1e6)*f
zeta   = 2.0  # constant in source term G at heat equation
kD     = 1e-6 # constant at the convective velocity v; conc equation
D0     = 5.0  # diffusion constant; conc equation 
kappa = kappaa/( rhoa*Ca) # diffusion; heat equation
nu    = rhob*Cb/(rhoa*Ca) # perfusion factor; heat equation
gamma = 0.85 # Newmark parameter
beta  = 0.45 # Newmark parameter
alpha_caputo = 0.8 # nonlocal time derivative exponent
max_fpi = 30    # maximum number of linear iterations for each time step
tol     = 1e-12 # relative error allowed to approve convergence of a step
vel_cte = ufl.as_vector((0.0,0.0,0.0)) # fixed convective velocity v0

## Functions:
c_speed = lambda ss: 1529.3 + 1.6856*ss + 6.1131e-2*ss**2 - 2.2967e-3*ss**3 + 2.2657e-5*ss**4 - 7.1795e-8*ss**5
qq = lambda ss: c_speed(ss)**2
kk = lambda ss: betaacou/(rho*qq(ss))
bb = lambda ss: 2.0*alpha*(c_speed(ss)**3)/(omega*omega)
#--------------------------------------------------------
omg_b = lambda ss: 0.0005 + 0.0001*ss
GG    = lambda pt, ss: zeta/(rhoa*Ca)*(bb(ss)/(qq(ss)*qq(ss)))*pt*pt
#--------------------------------------------------------
vv_drug = lambda p: vel_cte - kD*ufl.grad(p)
DD_drug = lambda p, ss: D0
#===============================================
# Timestepping:
#-----------------------------------------------
tau = 1e-4/1500
T   = 1500*tau
t   = 0
inc = 0
#===============================================
# Mesh creation
#domain   = mesh.create_rectangle(MPI.COMM_WORLD, mysquare,[40, 40], mesh.CellType.triangle)
with XDMFFile(MPI.COMM_WORLD, "mesh-half-circle-square.xdmf", "r") as file0:
    domain = file0.read_mesh(name = "Grid")
#======================================================================================================
# Vector spaces
ell = 1 
Hh = fem.FunctionSpace(domain, ("CG", ell))
Qh = fem.FunctionSpace(domain, ("CG", ell)) 
Wh = fem.FunctionSpace(domain, ("CG", ell))
#=================================================================
# Trial functions
# uu:  second order time derivative of pressure p; 
# zz:  temperature theta; 
# psi: concentration c.
uu  = ufl.TrialFunction(Hh)
zz  = ufl.TrialFunction(Qh)
psi = ufl.TrialFunction(Wh)
#--------------------------
# Test function
v   = ufl.TestFunction(Hh)
phi = ufl.TestFunction(Qh)
tts = ufl.TestFunction(Wh)
#===================================================================================
# Unknowns (pp,dp, ddp) and (theta) and auxiliar variable for fixed point iteration
pp  = fem.Function(Hh)
dp  = fem.Function(Hh)
ddp = fem.Function(Hh)
ddp_aux = fem.Function(Hh)

theta = fem.Function(Qh)
conc  = fem.Function(Wh)
ff = fem.Function(Hh)
gg = fem.Function(Hh)
#------------------------------------------------------------------------
# Newmark and nonlocal variables:
pp_pred   = fem.Function(Hh)
dp_pred   = fem.Function(Hh)
dp_caputo = fem.Function(Hh)
#---------------------------
theta_old = fem.Function(Qh)
conc_old  = fem.Function(Wh)
uNg = fem.Function(Hh)

#======================================================================== 
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
###======================================================================
def boundaryfunction(tt):
    # function g(t)
    wt = wfrec*tt
    SN, SN4 = np.sin(wt), np.sin(wt/4.0)
    CN, CN4 = np.cos(wt), np.cos(wt/4.0)
    if tt > 2.0*np.pi/wfrec:
        f_bdry   = SN*( 1.0  +  SN4 )
        df_bdry  = wfrec*( CN*(1.0 + SN4 ) + (1.0/4.0)*SN*CN4 )
        ddf_bdry = ((wfrec**2)/16.0)*(8.0*CN*CN4 - 16.0*SN*( 1.0 + SN4 ) - SN*SN4)                
    else:
        f_bdry   =  SN
        df_bdry  =  wfrec*CN
        ddf_bdry = -(wfrec**2)*SN
    return gg0*f_bdry, gg0*df_bdry, gg0*ddf_bdry

def boundarytag(mydomain,fdimension,conditions,mymarks):
    # function to mark pieces of boundary 
    myfacets, f_idx, f_mrk = [], [], []  
    for i in range(len(mymarks)):
        facet0 = mesh.locate_entities_boundary(mydomain, fdimension, marker = conditions[i])
        f_idx.append(facet0)
        f_mrk.append(np.full_like(facet0,  mymarks[i]))
        myfacets.append(facet0)
    f_idx = np.hstack(f_idx).astype(np.int32)
    f_mrk = np.hstack(f_mrk).astype(np.int32)
    f_sorted = np.argsort(f_idx)
    f_tags = meshtags(mydomain, fdimension, f_idx[f_sorted], f_mrk[f_sorted])
    return f_tags, myfacets

facet_tag, facets_boundary = boundarytag(domain, fdim, [lambda x: x[1] <= 0.0], [2])
dofs_bottom = fem.locate_dofs_topological(Hh, fdim, facets_boundary[0])
ds = ufl.Measure("ds", domain, subdomain_data = facet_tag)

###======================================================================
theta_old.interpolate(lambda x: 0.0 + 0*x[0])    
theta.interpolate(lambda x:     0.0 + 0*x[0])

conc_old.interpolate(lambda x: 0.0 + 0*x[0])
conc.interpolate(lambda x:     0.0 + 0*x[0])

pp.interpolate( lambda x: 0*x[0])
dp.interpolate( lambda x: 0*x[0])
ddp.interpolate(lambda x: 0*x[0])
dp_caputo.interpolate(lambda x: 0*x[0])

#------------------------------------------------------------------------------------------------
# Coefficients nonlocal term
zeta0 = 1.0/(2.0*math.gamma(2.0 - alpha_caputo))
zetaj = lambda ii: zeta0*((ii + 1.0)**(1.0 - alpha_caputo) - (ii-1.0)**(1.0 - alpha_caputo))
#------------------------------------------------------------------------------------------------
## Commented is the version after dividing by B(\Theta)
#CC1 = lambda ss, p_ast: (1.0 - 2.0*kk(ss + thetaa)*p_ast)/bb(ss + thetaa)                          ## --> (***)/B(Theta)
#CC2 = lambda ss: tau*tau*(beta*qq(ss + thetaa)/bb(ss + thetaa) + gamma*zeta0*tau**(-alpha_caputo)) ## --> (***)/B(Theta)
#Q   = lambda ss: qq(ss + thetaa)/bb(ss + thetaa)                                                   ## --> (***)/B(Theta)
#FF  = lambda ss, pt: 2.0*kk(ss + thetaa)*pt*pt/bb(ss + thetaa)                                     ## --> (***)/B(Theta)

CC1 = lambda ss, p_ast: (1.0 - 2.0*kk(ss + thetaa)*p_ast)
CC2 = lambda ss: tau*tau*(beta*qq(ss + thetaa) + gamma*bb(ss + thetaa)*zeta0*tau**(-alpha_caputo))
Q   = lambda ss: qq(ss + thetaa)
FF  = lambda ss, pt: 2.0*kk(ss + thetaa)*pt*pt
#------------------------------------------------------------------------------------------------
## to define and add source terms ffp, fftheta, ffcc
taualph = tau**(1 - alpha_caputo)

##-----------------------------------------------------------------------------------------------
### Spatial operators:
#def operators_wave_Newmark(tt): ## --> (***)/B(Theta)
#    a_wave  = CC1(theta,pp)*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(CC2(theta)*v))*ufl.dx
#    f_wave = (FF(theta,dp)*v - ufl.dot(ufl.grad(pp_pred), ufl.grad(Q(theta)*v)))*ufl.dx
#    f_wave += -taualph*ufl.dot(ufl.grad(dp_caputo),ufl.grad(v))*ufl.dx   
#    f_wave += -taualph*zeta0*ufl.dot(ufl.grad(dp_pred),ufl.grad(v))*ufl.dx   
#    return a_wave, f_wave       ## --> (***)/B(Theta)

def operators_wave_Newmark(tt):
    a_wave  = CC1(theta,pp)*uu*v*ufl.dx + ufl.dot(ufl.grad(uu),ufl.grad(CC2(theta)*v))*ufl.dx
    f_wave = (FF(theta,dp)*v - ufl.dot(ufl.grad(pp_pred), ufl.grad(Q(theta)*v)))*ufl.dx
    f_wave += -taualph*ufl.dot(ufl.grad(dp_caputo),ufl.grad(bb(theta + thetaa)*v))*ufl.dx
    f_wave += -taualph*zeta0*ufl.dot(ufl.grad(dp_pred),ufl.grad(bb(theta + thetaa)*v))*ufl.dx
    return a_wave, f_wave
    
def operators_heat(tt):
    a_heat = zz*phi*ufl.dx + tau*kappa*ufl.dot(ufl.grad(zz), ufl.grad(phi))*ufl.dx\
             + tau*nu*omg_b(theta + thetaa)*zz*phi*ufl.dx
    f_heat = theta_old*phi*ufl.dx + tau*GG(dp, theta + thetaa)*phi*ufl.dx
    return a_heat, f_heat

def operators_mass(tt):  
    a_mass = psi*tts*ufl.dx
    a_mass += tau*ufl.dot(DD_drug(pp,theta)*ufl.grad(psi),ufl.grad(tts))*ufl.dx
    a_mass += tau*ufl.dot(psi*(vv_drug(pp)), ufl.grad(tts))*ufl.dx    
    f_mass = conc_old*tts*ufl.dx
    return a_mass, f_mass

def time_iteration_Newmark(t, ddp, dp, pp, pp_pred, dp_pred):
    error    = tol + 1.0
    iter_fpi = 0    
    valN, dtvalN, ddtvalN = boundaryfunction(t)
    #uNg.interpolate(lambda x: (qq(thetaa)/bb(thetaa))*valN + dtvalN + 0*x[0]) ## --> (***)/B(Theta)
    uNg.interpolate(lambda x: qq(thetaa)*valN + bb(thetaa)*dtvalN + 0*x[0])
    pp_pred.x.array[:] = pp.x.array + tau*dp.x.array + tau*tau*(0.5 - beta)*ddp.x.array
    dp_pred.x.array[:] = dp.x.array + (1 - gamma)*tau*ddp.x.array
    
    pp.x.array[:] = pp_pred.x.array
    dp.x.array[:] = dp_pred.x.array
    ddp_aux.x.array[:] = ddp.x.array
    
    while error > tol and iter_fpi < max_fpi:
        iter_fpi += 1
        L_wave, rhs_wave = operators_wave_Newmark(t)                 
        rhs_wave += uNg*v*ds(2)
        wave_problem = dfpet.LinearProblem(L_wave, rhs_wave, bcs = [])
        ddp = wave_problem.solve()
        pp.x.array[:] = pp_pred.x.array + beta*tau*tau*ddp.x.array
        dp.x.array[:] = dp_pred.x.array + gamma*tau*ddp.x.array
        
        normpp = np.sqrt(fem.assemble_scalar(fem.form(ddp*ddp*ufl.dx))) + DOLFIN_EPS        
        error  = np.sqrt(fem.assemble_scalar(fem.form((ddp_aux - ddp)**2*ufl.dx)))/normpp     
        # Save value to compute the difference ddp-norm between k and k+1    
        ddp_aux.x.array[:] = ddp.x.array
    return ddp, dp, pp, pp_pred, dp_pred
##====================================================================

frec_save = 1
addfolder = ""
listvecs  = []

with io.XDMFFile(domain.comm, "SIM/" + addfolder + "wave-heat-mass.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    opcaputo0 = dp.x.array       
    while t <= T:
        inc += 1
        t   += tau  
        
        print('time:', t, ' --- iteration: ', inc)                
        # Reset error and iter_fpi needed for fixed-point iterations (fpi)
        
        #----------------------------------------------------------------------------------------
        ## Solve wave equation at t^(n+1)
        dp_caputo.x.array[:] = dp.x.array                      
        opcaputo = zeta0*(inc**(1-alpha_caputo) - (inc-1)**(1-alpha_caputo))*opcaputo0
        for j in range(len(listvecs)):
            opcaputo = opcaputo + zetaj(j+1)*listvecs[j]
        opcaputo = opcaputo + zetaj(0)*dp_pred.x.array
        dp_caputo.x.array[:] = opcaputo
        ddp, dp, pp, pp_pred, dp_pred = time_iteration_Newmark(t, ddp, dp, pp, pp_pred, dp_pred)
        #----------------------------------------------------------------------------------------
        ## Solve heat equation at t^(n+1)
        L_heat, rhs_heat = operators_heat(t)
        heat_problem = dfpet.LinearProblem(L_heat, rhs_heat, bcs = [])
        theta = heat_problem.solve()
        theta_old.x.array[:] = theta.x.array
        #----------------------------------------------------------------------------------------
        ## Solve mass equation at t^(n+1)
        L_mass, rhs_mass = operators_mass(t)
        rhs_mass     +=  tau*0.01*tts*ds(2)
        mass_problem  = dfpet.LinearProblem(L_mass, rhs_mass, bcs = [])
        conc = mass_problem.solve()        
        conc_old.x.array[:] = conc.x.array 
        #----------------------------------------------------------------------------------------
               
        if inc % frec_save == 0: 
            #===================================================
            # Save to xmdf file:
            pp.name    = "pressure";      xdmf.write_function(pp,inc)
            theta.name = "temperature";   xdmf.write_function(theta,inc)
            conc.name  = "concentration"; xdmf.write_function(conc,inc)
            
        ## Nonlocal term: Save previous \partial_t p^n to compute the Caputo derivative
        listvecs.append(dp.x.array)
        
