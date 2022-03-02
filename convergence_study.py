# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:37:37 2022

@author: Nour DAHMEN
"""

### Modules

#Scientific computing modules
import numpy as np
import sympy as sym
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve import spsolve
from scipy.interpolate import interp2d
from scipy.integrate import simps

#Code acceleration module
import numba

#Plotting module
import matplotlib.pyplot as plt

### Symbolic computation
x, y, t= sym.symbols('x y t')

#Symbolic diffusion tensor
alpha=1e-9
Dxx_1=(y**2+(alpha)*x**2)/(x**2+y**2)
Dyy_1=((alpha)*y**2+x**2)/(x**2+y**2)
Dyx_1=-(1.0-(alpha))*x*y/(x**2+y**2)

#Symbolic analytical solution
fs_1=(sym.sin(sym.pi*x))*(sym.sin(sym.pi*y))
#Symbolic derivaties
dfsdx_1=sym.diff(fs_1,x)
dfsdy_1=sym.diff(fs_1,y)
#Symbolic analytical source 
sos_1=-sym.diff(Dxx_1*sym.diff(fs_1,x)+Dyx_1*sym.diff(fs_1,y),x)-sym.diff(Dyy_1*sym.diff(fs_1,y)+Dyx_1*sym.diff(fs_1,x),y)

#Conversion from symbolic to functions
sos=sym.lambdify((x,y),sos_1,"numpy")
Dxx_2=sym.lambdify((x,y),Dxx_1,"numpy")
Dyy_2=sym.lambdify((x,y),Dyy_1,"numpy")
Dyx_2=sym.lambdify((x,y),Dyx_1,"numpy")
fs=sym.lambdify((x,y),fs_1,"numpy")

### Functions 
@numba.jit(nopython=True)
def term_NLTPFA(Daa1,Daa2,Dab1,Dab2,ha,hb,qa1,qa2,Nx,Ny,i,j,car_f,U,ind1,ind2,ind21,ind4,ind41):
    '''
    Generate the discretization term of the NLTPFA scheme for the surface car_f
    '''
        
    a1=0.0
    a2=0.0
    
    nu1=0.0
    nu2=0.0
    
    d0=0.0
    d1=0.0
    
    tau1=Daa1/ha
    tau2=Dab1/qa1
    tau21=-Dab1/qa2
    
    tau3=Daa2/hb
    tau4=Dab2/qa2
    tau41=-Dab2/qa1
    
    if (i==Nx and car_f=='S1') or (i==1 and car_f=='S3') or (j==1 and car_f=='S4') or (j==Ny and car_f=='S2'):
        
        a1=U[ind2]*sgn(Dab1)*(tau2)+sgn(-Dab1)*(U[ind21])*tau21
        a2=U[ind4]*sgn(Dab2)*(tau4)+sgn(-Dab2)*(U[ind41])*tau41
        
        (nu1,nu2)=calc_nu(a1,a2)
            
        d0=-nu1*(tau1+sgn(Dab1)*(tau2)+sgn(-Dab1)*tau21)-nu2*tau3
        d1=nu2*(tau3+sgn(Dab2)*(tau4)+sgn(-Dab2)*tau41)+nu1*tau1
        
    else:
        
        a1=U[ind1]*tau1+U[ind2]*sgn(Dab1)*(tau2)+sgn(-Dab1)*(U[ind21])*tau21
        a2=U[ind1]*tau3+U[ind4]*sgn(Dab2)*(tau4)+sgn(-Dab2)*(U[ind41])*tau41
    
        (nu1,nu2)=calc_nu(a1,a2)
                
        d0=-nu1*(tau1+sgn(Dab1)*(tau2)+sgn(-Dab1)*tau21)
        d1=nu2*(tau3+sgn(Dab2)*(tau4)+sgn(-Dab2)*tau41)
        
    return((d0,d1))
    
    
@numba.jit(nopython=True)
def Matrix_NLTPFA(Dxx,Dyy,Dyx,x_all,y_all,mass,U):
    '''
    Generate the diagonals of the NLTPFA scheme discretization matrix
    '''    
    diag0=[]
    diag1=[]
    diag_1=[]
    
    diag10=[]
    
    diag_10=[]
            
    Nx=len(x_all[1:-1:2])
    Ny=len(y_all[1:-1:2])
    
    h1_all=np.array([x_all[2*i]-x_all[2*i-2] for i in range(1,Nx+1,1)])
    q1_all=np.array([y_all[2*j]-y_all[2*j-2] for j in range(1,Ny+1,1)])
    
    h2_all=np.array([x_all[2*i]-x_all[2*i-1] for i in range(1,Nx+1,1)])
    h3_all=np.array([x_all[2*i-1]-x_all[2*i-2] for i in range(1,Nx+1,1)])
    h4_all=np.array([x_all[2*i+1]-x_all[2*i] for i in range(1,Nx,1)]+[x_all[-1]-x_all[-2]])
    h5_all=np.array([x_all[1]-x_all[0]]+[x_all[2*i-2]-x_all[2*i-3] for i in range(2,Nx+1,1)])
    
    q2_all=np.array([y_all[2*j]-y_all[2*j-1] for j in range(1,Ny+1,1)])
    q3_all=np.array([y_all[2*j-1]-y_all[2*j-2] for j in range(1,Ny+1,1)])
    q4_all=np.array([y_all[2*j+1]-y_all[2*j] for j in range(1,Ny,1)]+[y_all[-1]-y_all[-2]])
    q5_all=np.array([y_all[1]-y_all[0]]+[y_all[2*j-2]-y_all[2*j-3] for j in range(2,Ny+1,1)])
    
    h1=0.0
    h2=0.0
    h3=0.0
    h4=0.0
    h5=0.0
    
    q1=0.0
    q2=0.0
    q3=0.0
    q4=0.0
    q5=0.0
    
    Daa=0.0
    Dab=0.0
    Dbb=0.0
    Dba=0.0
    
    Daa1=0.0
    Dab1=0.0
    Dbb1=0.0
    Dba1=0.0
    
    Daa_1=0.0
    Dab_1=0.0
    Dbb_1=0.0
    Dba_1=0.0
    
    ind1_S1=(0,0)
    ind2_S1=(0,0)
    ind21_S1=(0,0)
    ind4_S1=(0,0)
    ind41_S1=(0,0)
    
    ind1_S2=(0,0)
    ind2_S2=(0,0)
    ind21_S2=(0,0)
    ind4_S2=(0,0)
    ind41_S2=(0,0)
    
    ind1_S3=(0,0)
    ind2_S3=(0,0)
    ind21_S3=(0,0)
    ind4_S3=(0,0)
    ind41_S3=(0,0)
    
    ind1_S4=(0,0)
    ind2_S4=(0,0)
    ind21_S4=(0,0)
    ind4_S4=(0,0)
    ind41_S4=(0,0)
    
    term_S1=(0.0,0.0)
    term_S2=(0.0,0.0)
    term_S3=(0.0,0.0)
    term_S4=(0.0,0.0)
    
    A=0.0
    B=0.0
    C=0.0
    D=0.0
    
    for j in range(1,Ny+1,1):
        for i in range(1,Nx+1,1):
            
            h1=h1_all[i-1]
            h2=h2_all[i-1]
            h3=h3_all[i-1]
            h4=h4_all[i-1]
            h5=h5_all[i-1]
            
            q1=q1_all[j-1]
            q2=q2_all[j-1]
            q3=q3_all[j-1]
            q4=q4_all[j-1]
            q5=q5_all[j-1]
                        
            Daa=Dxx[2*j-1,2*i-1]
            Dab=Dyx[2*j-1,2*i-1]
            Dbb=Dyy[2*j-1,2*i-1]
            Dba=Dyx[2*j-1,2*i-1]
            
            ind1_S1=(2*j-1,2*i)
            ind2_S1=(2*j,2*i-1)
            ind21_S1=(2*j-2,2*i-1)
            
            ind1_S2=(2*j,2*i-1)
            ind2_S2=(2*j-1,2*i)
            ind21_S2=(2*j-1,2*i-2)
            
            ind1_S3=(2*j-1,2*i-2)
            ind2_S3=(2*j-2,2*i-1)
            ind21_S3=(2*j,2*i-1)
            
            ind1_S4=(2*j-2,2*i-1)
            ind2_S4=(2*j-1,2*i-2)
            ind21_S4=(2*j-1,2*i)

            if i==Nx:
                Daa1=Dxx[2*j-1,2*i]
                Dab1=Dyx[2*j-1,2*i]
                
                ind4_S1=(2*j-2,2*i)
                ind41_S1=(2*j,2*i)
            else:
                Daa1=Dxx[2*j-1,2*i+1]
                Dab1=Dyx[2*j-1,2*i+1]
                
                ind4_S1=(2*j-2,2*i+1)
                ind41_S1=(2*j,2*i+1)
                
            if i==1:
                Daa_1=Dxx[2*j-1,2*i-2]
                Dab_1=Dyx[2*j-1,2*i-2]
                
                ind4_S3=(2*j,2*i-2)
                ind41_S3=(2*j-2,2*i-2)
                
            else:
                Daa_1=Dxx[2*j-1,2*i-3]
                Dab_1=Dyx[2*j-1,2*i-3]
                
                ind4_S3=(2*j,2*i-3)
                ind41_S3=(2*j-2,2*i-3)
            
            if j==Ny:
                Dbb1=Dyy[2*j,2*i-1]
                Dba1=Dyx[2*j,2*i-1]
                
                ind4_S2=(2*j,2*i-2)
                ind41_S2=(2*j,2*i)
            else:
                Dbb1=Dyy[2*j+1,2*i-1]
                Dba1=Dyx[2*j+1,2*i-1]
                
                ind4_S2=(2*j+1,2*i-2)
                ind41_S2=(2*j+1,2*i)
                
            if j==1:
                Dbb_1=Dyy[2*j-2,2*i-1]
                Dba_1=Dyx[2*j-2,2*i-1]
                
                ind4_S4=(2*j-2,2*i)
                ind41_S4=(2*j-2,2*i-2)  
            else:
                Dbb_1=Dyy[2*j-3,2*i-1]
                Dba_1=Dyx[2*j-3,2*i-1]
                
                ind4_S4=(2*j-3,2*i)
                ind41_S4=(2*j-3,2*i-2)         
                                        
            A=0.0
            B=0.0
            C=0.0
            D=0.0
            
            term_S1=term_NLTPFA(Daa,Daa1,Dab,Dab1,h2,h4,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1)
            term_S2=term_NLTPFA(Dbb,Dbb1,Dba,Dba1,q2,q4,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2)
            term_S3=term_NLTPFA(Daa,Daa_1,Dab,Dab_1,h3,h5,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3)
            term_S4=term_NLTPFA(Dbb,Dbb_1,Dba,Dba_1,q3,q5,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4)

            A=q1*term_S1[0]
            B=q1*term_S3[0]
            C=h1*term_S2[0]
            D=h1*term_S4[0]
            
            diag0.append(A+B+C+D)
            
            if i<Nx:
                
                A=0.0
                A=q1*term_S1[1]
    
                diag1.append(A)

            if i>1:
                
                B=0.0
                B=q1*term_S3[1]
                
                diag_1.append(B)
                
            if j<Ny:
                
                C=0.0
                C=h1*term_S2[1]
    
                diag10.append(C)
                    
            if j>1:
                
                D=0.0
                D=h1*term_S4[1]
                
                diag_10.append(D)
                
                            
        if j<Ny:
            diag1.append(0.0)
            diag_1.append(0.0)
            
    diago0=np.array(diag0)
    diago1=np.array([0]+diag1)
    diago_1=np.array(diag_1+[0])
    
    diago10=np.array([0]*(Nx)+diag10)
    
    diago_10=np.array(diag_10+[0]*(Nx))
    
    return((diago_10,diago_1,diago0,diago1,diago10))
    
    
    
def BC_NLTPFA(Dxx,Dyy,Dyx,x_all,y_all,fs,U):
    '''
    Generate NLTPFA scheme BC vector
    '''

    #Type Dirichlet
    mur_b=fs(x_all,y_all[0])    
    mur_d=fs(x_all[-1],y_all)   
    mur_h=fs(x_all,y_all[-1])   
    mur_g=fs(x_all[0],y_all) 
            
    s=0
    
    Nx=len(x_all[1:-1:2])
    Ny=len(y_all[1:-1:2])
        
    S1=np.zeros(Nx*Ny)
    
    h1=0.0
    h2=0.0
    h3=0.0
    h4=0.0
    h5=0.0
    
    q1=0.0
    q2=0.0
    q3=0.0
    q4=0.0
    q5=0.0
    
    Daa=0.0
    Dab=0.0
    Dbb=0.0
    Dba=0.0
    
    Daa1=0.0
    Dab1=0.0
    Dbb1=0.0
    Dba1=0.0
    
    Daa_1=0.0
    Dab_1=0.0
    Dbb_1=0.0
    Dba_1=0.0
    
    ind1_S1=(0,0)
    ind2_S1=(0,0)
    ind21_S1=(0,0)
    ind4_S1=(0,0)
    ind41_S1=(0,0)
    
    ind1_S2=(0,0)
    ind2_S2=(0,0)
    ind21_S2=(0,0)
    ind4_S2=(0,0)
    ind41_S2=(0,0)
    
    ind1_S3=(0,0)
    ind2_S3=(0,0)
    ind21_S3=(0,0)
    ind4_S3=(0,0)
    ind41_S3=(0,0)
    
    ind1_S4=(0,0)
    ind2_S4=(0,0)
    ind21_S4=(0,0)
    ind4_S4=(0,0)
    ind41_S4=(0,0)
        
    A=0.0
    
    term_S1=(0.0,0.0)
    term_S2=(0.0,0.0)
    term_S3=(0.0,0.0)
    term_S4=(0.0,0.0)

    for j in range(1,Ny+1,1):
        for i in range(1,Nx+1,1):
           
            if i==1:
                
                h1=x_all[2*i]-x_all[2*i-2]
                q1=y_all[2*j]-y_all[2*j-2] 

                h3=x_all[2*i-1]-x_all[2*i-2]
                h5=x_all[2*i-1]-x_all[2*i-2]
                
                q2=y_all[2*j]-y_all[2*j-1]
                q3=y_all[2*j-1]-y_all[2*j-2]
                
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                
                ind1_S3=(2*j-1,2*i-2)
                ind2_S3=(2*j-2,2*i-1)
                ind21_S3=(2*j,2*i-1)
                
                ind4_S3=(2*j,2*i-2)
                ind41_S3=(2*j-2,2*i-2)
                    
                Daa_1=Dxx[2*j-1,2*i-2]
                Dab_1=Dyx[2*j-1,2*i-2]
                
                term_S3=term_NLTPFA(Daa,Daa_1,Dab,Dab_1,h3,h5,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3)
                
                A=q1*term_S3[1]
                
                S1[s]=S1[s]+(A)*mur_g[2*j-1]
                
                
            if j==1:
                
                h1=x_all[2*i]-x_all[2*i-2]
                q1=y_all[2*j]-y_all[2*j-2] 
                
                h2=x_all[2*i]-x_all[2*i-1]
                h3=x_all[2*i-1]-x_all[2*i-2]
                
                q3=y_all[2*j-1]-y_all[2*j-2]
                q5=y_all[2*j-1]-y_all[2*j-2]
                
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
                
                ind1_S4=(2*j-2,2*i-1)
                ind2_S4=(2*j-1,2*i-2)
                ind21_S4=(2*j-1,2*i)
                    
                Dbb_1=Dyy[2*j-2,2*i-1]
                Dba_1=Dyx[2*j-2,2*i-1]
                
                ind4_S4=(2*j-2,2*i)
                ind41_S4=(2*j-2,2*i-2)  
                
                term_S4=term_NLTPFA(Dbb,Dbb_1,Dba,Dba_1,q3,q5,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4)
                                            
                A=h1*term_S4[1]
                
                S1[s]=S1[s]+(A)*mur_b[2*i-1]

            if i==Nx:
                
                h1=x_all[2*i]-x_all[2*i-2]
                q1=y_all[2*j]-y_all[2*j-2] 
                
                h2=x_all[2*i]-x_all[2*i-1]
                h4=x_all[2*i]-x_all[2*i-1]
                
                q2=y_all[2*j]-y_all[2*j-1]
                q3=y_all[2*j-1]-y_all[2*j-2]
                
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                
                ind1_S1=(2*j-1,2*i)
                ind2_S1=(2*j,2*i-1)
                ind21_S1=(2*j-2,2*i-1)
    
                Daa1=Dxx[2*j-1,2*i]
                Dab1=Dyx[2*j-1,2*i]
                
                ind4_S1=(2*j-2,2*i)
                ind41_S1=(2*j,2*i)
                
                term_S1=term_NLTPFA(Daa,Daa1,Dab,Dab1,h2,h4,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1)
                            
                A=q1*term_S1[1]
                
                S1[s]=S1[s]+A*mur_d[2*j-1]
            
            if j==Ny:
                
                h1=x_all[2*i]-x_all[2*i-2]
                q1=y_all[2*j]-y_all[2*j-2] 
                
                h2=x_all[2*i]-x_all[2*i-1]
                h3=x_all[2*i-1]-x_all[2*i-2]
                
                q2=y_all[2*j]-y_all[2*j-1]
                q4=y_all[2*j]-y_all[2*j-1]
                
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
                
                ind1_S2=(2*j,2*i-1)
                ind2_S2=(2*j-1,2*i)
                ind21_S2=(2*j-1,2*i-2)
                
                Dbb1=Dyy[2*j,2*i-1]
                Dba1=Dyx[2*j,2*i-1]
                
                ind4_S2=(2*j,2*i-2)
                ind41_S2=(2*j,2*i)
                
                term_S2=term_NLTPFA(Dbb,Dbb1,Dba,Dba1,q2,q4,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2)
                                                
                A=h1*term_S2[1]
                
                S1[s]=S1[s]+(A)*mur_h[2*i-1]
                
            s=s+1
            
    return(S1)
    
@numba.jit(nopython=True)
def term_NLMPFA(Daa1,Daa2,Dab1,Dab2,h,qa,qb,Nx,Ny,i,j,car_f,U,ind1,ind2,ind21,ind4,ind41):
    '''
    Generate the discretization term of the NLMPFA scheme for the surface car_f
    '''
    
    nu1=0.0
    nu2=0.0
    
    F1=0.0
    F2=0.0
    
    F11=0.0
    F22=0.0
        
    tau1=Daa1/h
    tau2=Dab1/qa
    tau21=-Dab1/qb
    tau3=Daa2/h
    tau4=Dab2/qb
    tau41=-Dab2/qa
        
    d0=0.0
    d01=0.0
    d0_1=0.0
    d1=0.0
    
    F1=tau1*(U[ind1]-U[2*j-1,2*i-1])+sgn(Dab1)*tau2*(U[ind2]-U[2*j-1,2*i-1])+sgn(-Dab1)*tau21*(U[ind21]-U[2*j-1,2*i-1])
    F2=tau3*(U[2*j-1,2*i-1]-U[ind1])+sgn(Dab2)*tau4*(U[ind4]-U[ind1])+sgn(-Dab2)*tau41*(U[ind41]-U[ind1])
    
    if F1*F2<=0.0:
        (nu1,nu2)=calc_nu(abs(F1),abs(F2))
        d0=-2*nu1*(tau1+sgn(Dab1)*tau2+sgn(-Dab1)*tau21)
        d1=2*nu1*tau1
        d01=2*nu1*sgn(Dab1)*tau2
        d0_1=2*nu1*sgn(-Dab1)*tau21
        
    elif F1*F2>0.0:
        
        F11=sgn(Dab1)*tau2*(U[ind2]-U[2*j-1,2*i-1])+sgn(-Dab1)*tau21*(U[ind21]-U[2*j-1,2*i-1])
        F22=sgn(Dab2)*tau4*(U[ind4]-U[ind1])+sgn(-Dab2)*tau41*(U[ind41]-U[ind1])
        
        if F11*F22<=0.0:
            (nu1,nu2)=calc_nu(abs(F11),abs(F22))
            d0=-2*nu1*(sgn(Dab1)*tau2+sgn(-Dab1)*tau21)-(tau1*nu1+tau3*nu2)
            d1=(tau1*nu1+tau3*nu2)
            d01=2*nu1*sgn(Dab1)*tau2
            d0_1=2*nu1*sgn(-Dab1)*tau21
            
        elif F11*F22>0.0:
            (nu1,nu2)=calc_nu(abs(F11),abs(F22))
            d0=-(tau1*nu1+tau3*nu2)
            d1=(tau1*nu1+tau3*nu2) 

    return((d0,d1,d01,d0_1)) 



@numba.jit(nopython=True)
def Matrix_NLMPFA(Dxx,Dyy,Dyx,x_all,y_all,mass,U):
    '''
    Generate the diagonals of the NLMPFA scheme discretization matrix
    '''    
    diag0=[]
    diag1=[]
    diag_1=[]
    
    diag10=[]
    
    diag_10=[]
        
    Nx=len(x_all[1:-1:2])
    Ny=len(y_all[1:-1:2])
        
    h1_all=np.array([x_all[2*i]-x_all[2*i-2] for i in range(1,Nx+1,1)])
    q1_all=np.array([y_all[2*j]-y_all[2*j-2] for j in range(1,Ny+1,1)])
    
    h2_all=np.array([x_all[2*i+1]-x_all[2*i-1] for i in range(1,Nx,1)]+[x_all[-1]-x_all[-2]])
    q2_all=np.array([y_all[2*j+1]-y_all[2*j-1] for j in range(1,Ny,1)]+[y_all[-1]-y_all[-2]])
    
    h3_all=np.array([x_all[1]-x_all[0]]+[x_all[2*i-1]-x_all[2*i-3] for i in range(2,Nx+1,1)])
    q3_all=np.array([y_all[1]-y_all[0]]+[y_all[2*j-1]-y_all[2*j-3] for j in range(2,Ny+1,1)])
    
    h1=0.0
    h2=0.0
    h3=0.0
    
    q1=0.0
    q2=0.0
    q3=0.0
    
    Daa=0.0
    Dab=0.0
    Dbb=0.0
    Dba=0.0
    
    Daa1=0.0
    Dab1=0.0
    Dbb1=0.0
    Dba1=0.0
    
    Daa_1=0.0
    Dab_1=0.0
    Dbb_1=0.0
    Dba_1=0.0
    
    ind1_S1=(0,0)
    ind2_S1=(0,0)
    ind21_S1=(0,0)
    ind4_S1=(0,0)
    ind41_S1=(0,0)
    
    ind1_S2=(0,0)
    ind2_S2=(0,0)
    ind21_S2=(0,0)
    ind4_S2=(0,0)
    ind41_S2=(0,0)
    
    ind1_S3=(0,0)
    ind2_S3=(0,0)
    ind21_S3=(0,0)
    ind4_S3=(0,0)
    ind41_S3=(0,0)
    
    ind1_S4=(0,0)
    ind2_S4=(0,0)
    ind21_S4=(0,0)
    ind4_S4=(0,0)
    ind41_S4=(0,0)
    
    term_S1=(0.0,0.0,0.0,0.0)
    term_S2=(0.0,0.0,0.0,0.0)
    term_S3=(0.0,0.0,0.0,0.0)
    term_S4=(0.0,0.0,0.0,0.0)
    
    A=0.0
    B=0.0
    C=0.0
    D=0.0

    
    for j in range(1,Ny+1,1):
        for i in range(1,Nx+1,1):
            
            h1=h1_all[i-1]
            h2=h2_all[i-1]
            h3=h3_all[i-1]
            
            q1=q1_all[j-1]
            q2=q2_all[j-1]
            q3=q3_all[j-1]
            
            Daa=Dxx[2*j-1,2*i-1]
            Dab=Dyx[2*j-1,2*i-1]
            Dbb=Dyy[2*j-1,2*i-1]
            Dba=Dyx[2*j-1,2*i-1]
            
            if i==Nx:
                ind1_S1=(2*j-1,2*i)
                Daa1=Dxx[2*j-1,2*i]
                Dab1=Dyx[2*j-1,2*i]
            else:
                ind1_S1=(2*j-1,2*i+1)
                Daa1=Dxx[2*j-1,2*i+1]
                Dab1=Dyx[2*j-1,2*i+1]
                
            if j==Ny:
                ind2_S1=(2*j,2*i-1)
            else:
                ind2_S1=(2*j+1,2*i-1)
                
            if j==1:
                ind21_S1=(2*j-2,2*i-1)
            else:
                ind21_S1=(2*j-3,2*i-1)
            
            if j==1:
                if i==Nx:
                    ind4_S1=(2*j-2,2*i)
                else:
                    ind4_S1=(2*j-2,2*i+1)
            elif i==Nx:
                ind4_S1=(2*j-3,2*i)
            else:
                ind4_S1=(2*j-3,2*i+1)
            
            if j==Ny:
                if i==Nx:
                    ind41_S1=(2*j,2*i)
                else:
                    ind41_S1=(2*j,2*i+1)
            elif i==Nx:
                ind41_S1=(2*j+1,2*i)
            else:
                ind41_S1=(2*j+1,2*i+1)
                  
            if j==Ny:
                ind1_S2=(2*j,2*i-1)
                Dbb1=Dyy[2*j,2*i-1]
                Dba1=Dyx[2*j,2*i-1]
            else:
                ind1_S2=(2*j+1,2*i-1)
                Dbb1=Dyy[2*j+1,2*i-1]
                Dba1=Dyx[2*j+1,2*i-1]
             
            if i==Nx:
                ind2_S2=(2*j-1,2*i)
            else:
                ind2_S2=(2*j-1,2*i+1)
            
            if i==1:
                ind21_S2=(2*j-1,2*i-2)
            else:
                ind21_S2=(2*j-1,2*i-3)
            
            if i==1:
                if j==Ny:
                    ind4_S2=(2*j,2*i-2)
                else:
                    ind4_S2=(2*j+1,2*i-2)
            elif j==Ny:
                ind4_S2=(2*j,2*i-3)
            else:         
                ind4_S2=(2*j+1,2*i-3)
                
            if i==Nx:
                if j==Ny:
                    ind41_S2=(2*j,2*i)
                else:
                    ind41_S2=(2*j+1,2*i)
            elif j==Ny:
                ind41_S2=(2*j,2*i+1)
            else:
                ind41_S2=(2*j+1,2*i+1)
                      
            if i==1:
                ind1_S3=(2*j-1,2*i-2)
                Daa_1=Dxx[2*j-1,2*i-2]
                Dab_1=Dyx[2*j-1,2*i-2]
            else:
                ind1_S3=(2*j-1,2*i-3)
                Daa_1=Dxx[2*j-1,2*i-3]
                Dab_1=Dyx[2*j-1,2*i-3]
            
            if j==1:
                ind2_S3=(2*j-2,2*i-1)
            else:
                ind2_S3=(2*j-3,2*i-1)
            
            if j==Ny:
                ind21_S3=(2*j,2*i-1)
            else:
                ind21_S3=(2*j+1,2*i-1)
            
            if j==Ny:
                if i==1:
                    ind4_S3=(2*j,2*i-2)
                else:
                    ind4_S3=(2*j,2*i-3)
            elif i==1:
                ind4_S3=(2*j+1,2*i-2)
            else:
                ind4_S3=(2*j+1,2*i-3)
            
            if j==1:
                if i==1:
                    ind41_S3=(2*j-2,2*i-2)
                else:
                    ind41_S3=(2*j-2,2*i-3)
            elif i==1:
                ind41_S3=(2*j-3,2*i-2)
            else:
                ind41_S3=(2*j-3,2*i-3)
                
            
            if j==1:
                ind1_S4=(2*j-2,2*i-1)
                Dbb_1=Dyy[2*j-2,2*i-1]
                Dba_1=Dyx[2*j-2,2*i-1]
            else:
                ind1_S4=(2*j-3,2*i-1)
                Dbb_1=Dyy[2*j-3,2*i-1]
                Dba_1=Dyx[2*j-3,2*i-1]
                
            if i==1:
                ind2_S4=(2*j-1,2*i-2)
            else:
                ind2_S4=(2*j-1,2*i-3)
            
            if i==Nx:
                ind21_S4=(2*j-1,2*i)
            else:
                ind21_S4=(2*j-1,2*i+1)
            
            if i==Nx:
                if j==1:
                    ind4_S4=(2*j-2,2*i)
                else:
                    ind4_S4=(2*j-3,2*i)
            elif j==1:
                ind4_S4=(2*j-2,2*i+1)
            else:
                ind4_S4=(2*j-3,2*i+1)
    
            if i==1:
                if j==1:
                    ind41_S4=(2*j-2,2*i-2)
                else:
                    ind41_S4=(2*j-3,2*i-2)
            elif j==1:
                ind41_S4=(2*j-2,2*i-3)
            else:
                ind41_S4=(2*j-3,2*i-3)
            
            A=0.0
            B=0.0
            C=0.0
            D=0.0
            
            term_S1=term_NLMPFA(Daa,Daa1,Dab,Dab1,h2,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1)
            term_S2=term_NLMPFA(Dbb,Dbb1,Dba,Dba1,q2,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2)
            term_S3=term_NLMPFA(Daa,Daa_1,Dab,Dab_1,h3,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3)
            term_S4=term_NLMPFA(Dbb,Dbb_1,Dba,Dba_1,q3,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4)

            A=q1*term_S1[0]
            B=q1*term_S3[0]
            C=h1*term_S2[0]
            D=h1*term_S4[0]
            
            diag0.append(A+B+C+D)
            
            if i<Nx:
                
                A=0.0
                C=0.0
                D=0.0
                
                A=q1*term_S1[1]
                C=h1*term_S2[2]
                D=h1*term_S4[3]
                    
                diag1.append(A+C+D)

            if i>1:
                
                B=0.0
                C=0.0
                D=0.0
                
                B=q1*term_S3[1]
                C=h1*term_S2[3]
                D=h1*term_S4[2]
                
                diag_1.append(B+C+D)
                
            if j<Ny:
                
                A=0.0
                B=0.0
                C=0.0
                
                A=q1*term_S1[2]
                B=q1*term_S3[3]
                C=h1*term_S2[1]
                
                diag10.append(A+B+C)
                
                    
            if j>1:
                
                A=0.0
                B=0.0
                D=0.0
                
                A=q1*term_S1[3]
                B=q1*term_S3[2]
                D=h1*term_S4[1]
                
                diag_10.append(A+B+D)
                            
        if j<Ny:
            diag1.append(0.0)
            diag_1.append(0.0)
            
        
    diago0=np.array(diag0)
    diago1=np.array([0]+(diag1))
    diago_1=np.array((diag_1)+[0])
    
    diago10=np.array([0]*(Nx)+(diag10))
    
    diago_10=np.array((diag_10)+[0]*(Nx))
    
    return((diago_10,diago_1,diago0,diago1,diago10))


def BC_NLMPFA(Dxx,Dyy,Dyx,x_all,y_all,fs,U):
    '''
    Generate NLMPFA scheme BC vector
    '''
    #Type Dirichlet
    mur_b=fs(x_all,y_all[0])    
    mur_d=fs(x_all[-1],y_all)   
    mur_h=fs(x_all,y_all[-1])   
    mur_g=fs(x_all[0],y_all)   
                
    s=0
    
    Nx=len(x_all[1:-1:2])
    Ny=len(y_all[1:-1:2])
        
    S1=np.zeros(Nx*Ny)
    
    h1=0.0
    h2=0.0
    h3=0.0
    
    q1=0.0
    q2=0.0
    q3=0.0
    
    Daa=0.0
    Dab=0.0
    Dbb=0.0
    Dba=0.0
    
    Daa1=0.0
    Dab1=0.0
    Dbb1=0.0
    Dba1=0.0
    
    Daa_1=0.0
    Dab_1=0.0
    Dbb_1=0.0
    Dba_1=0.0
    
    ind1_S1=(0,0)
    ind2_S1=(0,0)
    ind21_S1=(0,0)
    ind4_S1=(0,0)
    ind41_S1=(0,0)
    
    ind1_S2=(0,0)
    ind2_S2=(0,0)
    ind21_S2=(0,0)
    ind4_S2=(0,0)
    ind41_S2=(0,0)
    
    ind1_S3=(0,0)
    ind2_S3=(0,0)
    ind21_S3=(0,0)
    ind4_S3=(0,0)
    ind41_S3=(0,0)
    
    ind1_S4=(0,0)
    ind2_S4=(0,0)
    ind21_S4=(0,0)
    ind4_S4=(0,0)
    ind41_S4=(0,0)
        
    A=0.0
    B=0.0
    C=0.0
    D=0.0
    
    term_S1=(0.0,0.0,0.0,0.0)
    term_S2=(0.0,0.0,0.0,0.0)
    term_S3=(0.0,0.0,0.0,0.0)
    term_S4=(0.0,0.0,0.0,0.0)
    
    for j in range(1,Ny+1,1):
        for i in range(1,Nx+1,1):
            
            if i==1:
                
                h1=x_all[2*i]-x_all[2*i-2]# xi+1/2 - xi-1/2  
            
                h2=x_all[2*i+1]-x_all[2*i-1]# xi+1/2 - xi 
                h3=x_all[2*i-1]-x_all[2*i-2]# xi - xi-1/2
                
                q1=y_all[2*j]-y_all[2*j-2]# yj+1/2 - yj-1/2
                
                if j==Ny:
                    q2=y_all[2*j]-y_all[2*j-1]# yj+1/2- yj
                else:
                    q2=y_all[2*j+1]-y_all[2*j-1]# yj+1/2- yj
                
                if j==1:
                    q3=y_all[2*j-1]-y_all[2*j-2]# yj - yj-1/2
                else:
                    q3=y_all[2*j-1]-y_all[2*j-3]# yj - yj-1/2     
        
                
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
                
                      
                if j==Ny:
                    ind1_S2=(2*j,2*i-1)
                    Dbb1=Dyy[2*j,2*i-1]
                    Dba1=Dyx[2*j,2*i-1]
                else:
                    ind1_S2=(2*j+1,2*i-1)
                    Dbb1=Dyy[2*j+1,2*i-1]
                    Dba1=Dyx[2*j+1,2*i-1]
                 
                if i==Nx:
                    ind2_S2=(2*j-1,2*i)
                else:
                    ind2_S2=(2*j-1,2*i+1)
                
                if i==1:
                    ind21_S2=(2*j-1,2*i-2)
                else:
                    ind21_S2=(2*j-1,2*i-3)
                
                if i==1:
                    if j==Ny:
                        ind4_S2=(2*j,2*i-2)
                    else:
                        ind4_S2=(2*j+1,2*i-2)
                elif j==Ny:
                    ind4_S2=(2*j,2*i-3)
                else:         
                    ind4_S2=(2*j+1,2*i-3)
                    
                if i==Nx:
                    if j==Ny:
                        ind41_S2=(2*j,2*i)
                    else:
                        ind41_S2=(2*j+1,2*i)
                elif j==Ny:
                    ind41_S2=(2*j,2*i+1)
                else:
                    ind41_S2=(2*j+1,2*i+1)
                          
                          
                if i==1:
                    ind1_S3=(2*j-1,2*i-2)
                    Daa_1=Dxx[2*j-1,2*i-2]
                    Dab_1=Dyx[2*j-1,2*i-2]
                else:
                    ind1_S3=(2*j-1,2*i-3)
                    Daa_1=Dxx[2*j-1,2*i-3]
                    Dab_1=Dyx[2*j-1,2*i-3]
                
                if j==1:
                    ind2_S3=(2*j-2,2*i-1)
                else:
                    ind2_S3=(2*j-3,2*i-1)
                
                if j==Ny:
                    ind21_S3=(2*j,2*i-1)
                else:
                    ind21_S3=(2*j+1,2*i-1)
                
                if j==Ny:
                    if i==1:
                        ind4_S3=(2*j,2*i-2)
                    else:
                        ind4_S3=(2*j,2*i-3)
                elif i==1:
                    ind4_S3=(2*j+1,2*i-2)
                else:
                    ind4_S3=(2*j+1,2*i-3)
                
                if j==1:
                    if i==1:
                        ind41_S3=(2*j-2,2*i-2)
                    else:
                        ind41_S3=(2*j-2,2*i-3)
                elif i==1:
                    ind41_S3=(2*j-3,2*i-2)
                else:
                    ind41_S3=(2*j-3,2*i-3)
                    
                
                if j==1:
                    ind1_S4=(2*j-2,2*i-1)
                    Dbb_1=Dyy[2*j-2,2*i-1]
                    Dba_1=Dyx[2*j-2,2*i-1]
                else:
                    ind1_S4=(2*j-3,2*i-1)
                    Dbb_1=Dyy[2*j-3,2*i-1]
                    Dba_1=Dyx[2*j-3,2*i-1]
                    
                if i==1:
                    ind2_S4=(2*j-1,2*i-2)
                else:
                    ind2_S4=(2*j-1,2*i-3)
                
                if i==Nx:
                    ind21_S4=(2*j-1,2*i)
                else:
                    ind21_S4=(2*j-1,2*i+1)
                
                if i==Nx:
                    if j==1:
                        ind4_S4=(2*j-2,2*i)
                    else:
                        ind4_S4=(2*j-3,2*i)
                elif j==1:
                    ind4_S4=(2*j-2,2*i+1)
                else:
                    ind4_S4=(2*j-3,2*i+1)
        
                if i==1:
                    if j==1:
                        ind41_S4=(2*j-2,2*i-2)
                    else:
                        ind41_S4=(2*j-3,2*i-2)
                elif j==1:
                    ind41_S4=(2*j-2,2*i-3)
                else:
                    ind41_S4=(2*j-3,2*i-3)
                
                B=0.0
                C=0.0
                D=0.0
                
                term_S3=term_NLMPFA(Daa,Daa_1,Dab,Dab_1,h3,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3)
                term_S2=term_NLMPFA(Dbb,Dbb1,Dba,Dba1,q2,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2)
                term_S4=term_NLMPFA(Dbb,Dbb_1,Dba,Dba_1,q3,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4)
                    
                B=q1*term_S3[1]
                C=h1*term_S2[3]
                D=h1*term_S4[2]
                
                S1[s]=S1[s]+(B+C+D)*mur_g[2*j-1]
                
            if j==1:
                
                h1=x_all[2*i]-x_all[2*i-2]# xi+1/2 - xi-1/2  
            
                if i==Nx:
                    h2=x_all[2*i]-x_all[2*i-1]# xi+1/2 - xi 
                else:
                    h2=x_all[2*i+1]-x_all[2*i-1]# xi+1/2 - xi 
                
                if i==1:
                    h3=x_all[2*i-1]-x_all[2*i-2]# xi - xi-1/2
                else:
                    h3=x_all[2*i-1]-x_all[2*i-3]# xi - xi-1/2
                
                q1=y_all[2*j]-y_all[2*j-2]# yj+1/2 - yj-1/2
                q2=y_all[2*j+1]-y_all[2*j-1]# yj+1/2- yj
                q3=y_all[2*j-1]-y_all[2*j-2]# yj - yj-1/2
                
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
                
                if i==Nx:
                    ind1_S1=(2*j-1,2*i)
                    Daa1=Dxx[2*j-1,2*i]
                    Dab1=Dyx[2*j-1,2*i]
                else:
                    ind1_S1=(2*j-1,2*i+1)
                    Daa1=Dxx[2*j-1,2*i+1]
                    Dab1=Dyx[2*j-1,2*i+1]
                    
                if j==Ny:
                    ind2_S1=(2*j,2*i-1)
                else:
                    ind2_S1=(2*j+1,2*i-1)
                    
                if j==1:
                    ind21_S1=(2*j-2,2*i-1)
                else:
                    ind21_S1=(2*j-3,2*i-1)
                
                if j==1:
                    if i==Nx:
                        ind4_S1=(2*j-2,2*i)
                    else:
                        ind4_S1=(2*j-2,2*i+1)
                elif i==Nx:
                    ind4_S1=(2*j-3,2*i)
                else:
                    ind4_S1=(2*j-3,2*i+1)
                
                if j==Ny:
                    if i==Nx:
                        ind41_S1=(2*j,2*i)
                    else:
                        ind41_S1=(2*j,2*i+1)
                elif i==Nx:
                    ind41_S1=(2*j+1,2*i)
                else:
                    ind41_S1=(2*j+1,2*i+1)
                      
                          
                if i==1:
                    ind1_S3=(2*j-1,2*i-2)
                    Daa_1=Dxx[2*j-1,2*i-2]
                    Dab_1=Dyx[2*j-1,2*i-2]
                else:
                    ind1_S3=(2*j-1,2*i-3)
                    Daa_1=Dxx[2*j-1,2*i-3]
                    Dab_1=Dyx[2*j-1,2*i-3]
                
                if j==1:
                    ind2_S3=(2*j-2,2*i-1)
                else:
                    ind2_S3=(2*j-3,2*i-1)
                
                if j==Ny:
                    ind21_S3=(2*j,2*i-1)
                else:
                    ind21_S3=(2*j+1,2*i-1)
                
                if j==Ny:
                    if i==1:
                        ind4_S3=(2*j,2*i-2)
                    else:
                        ind4_S3=(2*j,2*i-3)
                elif i==1:
                    ind4_S3=(2*j+1,2*i-2)
                else:
                    ind4_S3=(2*j+1,2*i-3)
                
                if j==1:
                    if i==1:
                        ind41_S3=(2*j-2,2*i-2)
                    else:
                        ind41_S3=(2*j-2,2*i-3)
                elif i==1:
                    ind41_S3=(2*j-3,2*i-2)
                else:
                    ind41_S3=(2*j-3,2*i-3)
                    
                
                if j==1:
                    ind1_S4=(2*j-2,2*i-1)
                    Dbb_1=Dyy[2*j-2,2*i-1]
                    Dba_1=Dyx[2*j-2,2*i-1]
                else:
                    ind1_S4=(2*j-3,2*i-1)
                    Dbb_1=Dyy[2*j-3,2*i-1]
                    Dba_1=Dyx[2*j-3,2*i-1]
                    
                if i==1:
                    ind2_S4=(2*j-1,2*i-2)
                else:
                    ind2_S4=(2*j-1,2*i-3)
                
                if i==Nx:
                    ind21_S4=(2*j-1,2*i)
                else:
                    ind21_S4=(2*j-1,2*i+1)
                
                if i==Nx:
                    if j==1:
                        ind4_S4=(2*j-2,2*i)
                    else:
                        ind4_S4=(2*j-3,2*i)
                elif j==1:
                    ind4_S4=(2*j-2,2*i+1)
                else:
                    ind4_S4=(2*j-3,2*i+1)
        
                if i==1:
                    if j==1:
                        ind41_S4=(2*j-2,2*i-2)
                    else:
                        ind41_S4=(2*j-3,2*i-2)
                elif j==1:
                    ind41_S4=(2*j-2,2*i-3)
                else:
                    ind41_S4=(2*j-3,2*i-3)
                
                A=0.0
                B=0.0
                D=0.0
                
                term_S1=term_NLMPFA(Daa,Daa1,Dab,Dab1,h2,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1)
                term_S3=term_NLMPFA(Daa,Daa_1,Dab,Dab_1,h3,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3)
                term_S4=term_NLMPFA(Dbb,Dbb_1,Dba,Dba_1,q3,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4)
                
                A=q1*term_S1[3]
                B=q1*term_S3[2]
                D=h1*term_S4[1]
                
                S1[s]=S1[s]+(A+B+D)*mur_b[2*i-1]

            if i==Nx:
                
                h1=x_all[2*i]-x_all[2*i-2]# xi+1/2 - xi-1/2  
                h2=x_all[2*i]-x_all[2*i-1]# xi+1/2 - xi 
                h3=x_all[2*i-1]-x_all[2*i-3]# xi - xi-1/2
                
                q1=y_all[2*j]-y_all[2*j-2]# yj+1/2 - yj-1/2
                
                if j==Ny:
                    q2=y_all[2*j]-y_all[2*j-1]# yj+1/2- yj
                else:
                    q2=y_all[2*j+1]-y_all[2*j-1]# yj+1/2- yj
                
                if j==1:
                    q3=y_all[2*j-1]-y_all[2*j-2]# yj - yj-1/2
                else:
                    q3=y_all[2*j-1]-y_all[2*j-3]# yj - yj-1/2 
                    
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
                
                if i==Nx:
                    ind1_S1=(2*j-1,2*i)
                    Daa1=Dxx[2*j-1,2*i]
                    Dab1=Dyx[2*j-1,2*i]
                else:
                    ind1_S1=(2*j-1,2*i+1)
                    Daa1=Dxx[2*j-1,2*i+1]
                    Dab1=Dyx[2*j-1,2*i+1]
                    
                if j==Ny:
                    ind2_S1=(2*j,2*i-1)
                else:
                    ind2_S1=(2*j+1,2*i-1)
                    
                if j==1:
                    ind21_S1=(2*j-2,2*i-1)
                else:
                    ind21_S1=(2*j-3,2*i-1)
                
                if j==1:
                    if i==Nx:
                        ind4_S1=(2*j-2,2*i)
                    else:
                        ind4_S1=(2*j-2,2*i+1)
                elif i==Nx:
                    ind4_S1=(2*j-3,2*i)
                else:
                    ind4_S1=(2*j-3,2*i+1)
                
                if j==Ny:
                    if i==Nx:
                        ind41_S1=(2*j,2*i)
                    else:
                        ind41_S1=(2*j,2*i+1)
                elif i==Nx:
                    ind41_S1=(2*j+1,2*i)
                else:
                    ind41_S1=(2*j+1,2*i+1)
                      
                if j==Ny:
                    ind1_S2=(2*j,2*i-1)
                    Dbb1=Dyy[2*j,2*i-1]
                    Dba1=Dyx[2*j,2*i-1]
                else:
                    ind1_S2=(2*j+1,2*i-1)
                    Dbb1=Dyy[2*j+1,2*i-1]
                    Dba1=Dyx[2*j+1,2*i-1]
                 
                if i==Nx:
                    ind2_S2=(2*j-1,2*i)
                else:
                    ind2_S2=(2*j-1,2*i+1)
                
                if i==1:
                    ind21_S2=(2*j-1,2*i-2)
                else:
                    ind21_S2=(2*j-1,2*i-3)
                
                if i==1:
                    if j==Ny:
                        ind4_S2=(2*j,2*i-2)
                    else:
                        ind4_S2=(2*j+1,2*i-2)
                elif j==Ny:
                    ind4_S2=(2*j,2*i-3)
                else:         
                    ind4_S2=(2*j+1,2*i-3)
                    
                if i==Nx:
                    if j==Ny:
                        ind41_S2=(2*j,2*i)
                    else:
                        ind41_S2=(2*j+1,2*i)
                elif j==Ny:
                    ind41_S2=(2*j,2*i+1)
                else:
                    ind41_S2=(2*j+1,2*i+1)
                                          
                if j==1:
                    ind1_S4=(2*j-2,2*i-1)
                    Dbb_1=Dyy[2*j-2,2*i-1]
                    Dba_1=Dyx[2*j-2,2*i-1]
                else:
                    ind1_S4=(2*j-3,2*i-1)
                    Dbb_1=Dyy[2*j-3,2*i-1]
                    Dba_1=Dyx[2*j-3,2*i-1]
                    
                if i==1:
                    ind2_S4=(2*j-1,2*i-2)
                else:
                    ind2_S4=(2*j-1,2*i-3)
                
                if i==Nx:
                    ind21_S4=(2*j-1,2*i)
                else:
                    ind21_S4=(2*j-1,2*i+1)
                
                if i==Nx:
                    if j==1:
                        ind4_S4=(2*j-2,2*i)
                    else:
                        ind4_S4=(2*j-3,2*i)
                elif j==1:
                    ind4_S4=(2*j-2,2*i+1)
                else:
                    ind4_S4=(2*j-3,2*i+1)
        
                if i==1:
                    if j==1:
                        ind41_S4=(2*j-2,2*i-2)
                    else:
                        ind41_S4=(2*j-3,2*i-2)
                elif j==1:
                    ind41_S4=(2*j-2,2*i-3)
                else:
                    ind41_S4=(2*j-3,2*i-3)

                A=0.0
                C=0.0
                D=0.0
                
                term_S1=term_NLMPFA(Daa,Daa1,Dab,Dab1,h2,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1)
                term_S2=term_NLMPFA(Dbb,Dbb1,Dba,Dba1,q2,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2)
                term_S4=term_NLMPFA(Dbb,Dbb_1,Dba,Dba_1,q3,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4)
                                
                A=q1*term_S1[1]
                C=h1*term_S2[2]
                D=h1*term_S4[3] 

                S1[s]=S1[s]+(A+C+D)*mur_d[2*j-1]
            
            if j==Ny:
                
                h1=x_all[2*i]-x_all[2*i-2]# xi+1/2 - xi-1/2  
            
                if i==Nx:
                    h2=x_all[2*i]-x_all[2*i-1]# xi+1/2 - xi 
                else:
                    h2=x_all[2*i+1]-x_all[2*i-1]# xi+1/2 - xi 
                
                if i==1:
                    h3=x_all[2*i-1]-x_all[2*i-2]# xi - xi-1/2
                else:
                    h3=x_all[2*i-1]-x_all[2*i-3]# xi - xi-1/2
                
                q1=y_all[2*j]-y_all[2*j-2]# yj+1/2 - yj-1/2
                q2=y_all[2*j]-y_all[2*j-1]# yj+1/2- yj
                q3=y_all[2*j-1]-y_all[2*j-3]# yj - yj-1/2     
                
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
                
                if i==Nx:
                    ind1_S1=(2*j-1,2*i)
                    Daa1=Dxx[2*j-1,2*i]
                    Dab1=Dyx[2*j-1,2*i]
                else:
                    ind1_S1=(2*j-1,2*i+1)
                    Daa1=Dxx[2*j-1,2*i+1]
                    Dab1=Dyx[2*j-1,2*i+1]
                    
                if j==Ny:
                    ind2_S1=(2*j,2*i-1)
                else:
                    ind2_S1=(2*j+1,2*i-1)
                    
                if j==1:
                    ind21_S1=(2*j-2,2*i-1)
                else:
                    ind21_S1=(2*j-3,2*i-1)
                
                if j==1:
                    if i==Nx:
                        ind4_S1=(2*j-2,2*i)
                    else:
                        ind4_S1=(2*j-2,2*i+1)
                elif i==Nx:
                    ind4_S1=(2*j-3,2*i)
                else:
                    ind4_S1=(2*j-3,2*i+1)
                
                if j==Ny:
                    if i==Nx:
                        ind41_S1=(2*j,2*i)
                    else:
                        ind41_S1=(2*j,2*i+1)
                elif i==Nx:
                    ind41_S1=(2*j+1,2*i)
                else:
                    ind41_S1=(2*j+1,2*i+1)
                      
                if j==Ny:
                    ind1_S2=(2*j,2*i-1)
                    Dbb1=Dyy[2*j,2*i-1]
                    Dba1=Dyx[2*j,2*i-1]
                else:
                    ind1_S2=(2*j+1,2*i-1)
                    Dbb1=Dyy[2*j+1,2*i-1]
                    Dba1=Dyx[2*j+1,2*i-1]
                 
                if i==Nx:
                    ind2_S2=(2*j-1,2*i)
                else:
                    ind2_S2=(2*j-1,2*i+1)
                
                if i==1:
                    ind21_S2=(2*j-1,2*i-2)
                else:
                    ind21_S2=(2*j-1,2*i-3)
                
                if i==1:
                    if j==Ny:
                        ind4_S2=(2*j,2*i-2)
                    else:
                        ind4_S2=(2*j+1,2*i-2)
                elif j==Ny:
                    ind4_S2=(2*j,2*i-3)
                else:         
                    ind4_S2=(2*j+1,2*i-3)
                    
                if i==Nx:
                    if j==Ny:
                        ind41_S2=(2*j,2*i)
                    else:
                        ind41_S2=(2*j+1,2*i)
                elif j==Ny:
                    ind41_S2=(2*j,2*i+1)
                else:
                    ind41_S2=(2*j+1,2*i+1)
                          
                          
                          
                if i==1:
                    ind1_S3=(2*j-1,2*i-2)
                    Daa_1=Dxx[2*j-1,2*i-2]
                    Dab_1=Dyx[2*j-1,2*i-2]
                else:
                    ind1_S3=(2*j-1,2*i-3)
                    Daa_1=Dxx[2*j-1,2*i-3]
                    Dab_1=Dyx[2*j-1,2*i-3]
                
                if j==1:
                    ind2_S3=(2*j-2,2*i-1)
                else:
                    ind2_S3=(2*j-3,2*i-1)
                
                if j==Ny:
                    ind21_S3=(2*j,2*i-1)
                else:
                    ind21_S3=(2*j+1,2*i-1)
                
                if j==Ny:
                    if i==1:
                        ind4_S3=(2*j,2*i-2)
                    else:
                        ind4_S3=(2*j,2*i-3)
                elif i==1:
                    ind4_S3=(2*j+1,2*i-2)
                else:
                    ind4_S3=(2*j+1,2*i-3)
                
                if j==1:
                    if i==1:
                        ind41_S3=(2*j-2,2*i-2)
                    else:
                        ind41_S3=(2*j-2,2*i-3)
                elif i==1:
                    ind41_S3=(2*j-3,2*i-2)
                else:
                    ind41_S3=(2*j-3,2*i-3)
                    
                A=0.0
                B=0.0
                C=0.0
                
                term_S1=term_NLMPFA(Daa,Daa1,Dab,Dab1,h2,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1)
                term_S3=term_NLMPFA(Daa,Daa_1,Dab,Dab_1,h3,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3)
                term_S2=term_NLMPFA(Dbb,Dbb1,Dba,Dba1,q2,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2)
                                
                A=q1*term_S1[2]
                B=q1*term_S3[3]
                C=h1*term_S2[1]

                S1[s]=S1[s]+(A+B+C)*mur_h[2*i-1]
                
            s=s+1
            
    return(S1)
    
@numba.jit(nopython=True)
def term_RNLMPFA(Daa1,Daa2,Dab1,Dab2,h,qa,qb,Nx,Ny,i,j,car_f,U,ind1,ind2,ind21,ind4,ind41,als,bes):
    '''
    Generate the discretization term of the RNLMPFA scheme for the surface car_f
    '''
    c1=0.0
    c2=0.0
    
    G1=0.0
    G2=0.0
    
    nu1=0.0
    nu2=0.0

    tau1=Daa1/h
    tau2=(Dab1)/qa
    tau21=-(Dab1)/qb
    tau3=Daa2/h
    tau4=(Dab2)/qb
    tau41=-(Dab2)/qa
    
    
    if car_f=='S1' or car_f=='S2':
        c1=als
        c2=bes
    elif car_f=='S3' or car_f=='S4':
        c1=bes
        c2=als


    if i==1:
        if car_f=='S3' or car_f=='S2' or car_f=='S4':
            c1=0.0
            c2=0.0
    if i==Nx:
        if car_f=='S1' or car_f=='S2' or car_f=='S4':
            c1=0.0
            c2=0.0
    if j==1:
        if car_f=='S4' or car_f=='S1' or car_f=='S3':
            c1=0.0
            c2=0.0
    if j==Ny:
        if car_f=='S2' or car_f=='S1' or car_f=='S3':
            c1=0.0
            c2=0.0
      
    d0=0.0
    d01=0.0
    d0_1=0.0
    d1=0.0
    d_11=0.0
    d11=0.0
                
    G1=(1-c1)*(sgn(Dab1)*tau2*(U[ind2]-U[2*j-1,2*i-1])+sgn(-Dab1)*tau21*(U[ind21]-U[2*j-1,2*i-1]))
    G2=(1-c2)*(sgn(Dab2)*tau4*(U[ind4]-U[ind1])+sgn(-Dab2)*tau41*(U[ind41]-U[ind1]))
    
    if G1*G2<=0.0:
        
        (nu1,nu2)=calc_nu(abs(G1),abs(G2))
        
        d0=-(tau1*nu1+tau3*nu2)-(c1*nu1+2*nu1*(1.0-c1))*(sgn(Dab1)*tau2+sgn(-Dab1)*tau21)
        d1=(tau1*nu1+tau3*nu2)+c2*nu2*(sgn(Dab2)*tau4+sgn(-Dab2)*tau41)
        d01=(c1*nu1+2*nu1*(1.0-c1))*sgn(Dab1)*tau2
        d0_1=(c1*nu1+2*nu1*(1.0-c1))*sgn(-Dab1)*tau21
        d_11=-c2*nu2*(sgn(Dab2)*tau4)
        d11=-c2*nu2*(sgn(-Dab2)*tau41)
        
    elif G1*G2>0.0:
        (nu1,nu2)=calc_nu(abs(G1),abs(G2))
        
        d0=-(tau1*nu1+tau3*nu2)-c1*nu1*(sgn(Dab1)*tau2+sgn(-Dab1)*tau21)
        d1=(tau1*nu1+tau3*nu2)+c2*nu2*(sgn(Dab2)*tau4+sgn(-Dab2)*tau41) 
        d01=(c1*nu1)*sgn(Dab1)*tau2
        d0_1=(c1*nu1)*sgn(-Dab1)*tau21
        d_11=-c2*nu2*(sgn(Dab2)*tau4) 
        d11=-c2*nu2*(sgn(-Dab2)*tau41)
        
    return((d0,d1,d01,d0_1,d_11,d11)) 

@numba.jit(nopython=True)
def Matrix_RNLMPFA(Dxx,Dyy,Dyx,x_all,y_all,mass,U,als,bes):
    '''
    Generate the diagonals of the RNLMPFA scheme discretization matrix
    '''    
    diag0=[]
    diag1=[]
    diag_1=[]
    
    diag11=[]
    diag10=[]
    diag1_1=[0]
    
    diag_11=[0]
    diag_10=[]
    diag_1_1=[]
        
    Nx=len(x_all[1:-1:2])
    Ny=len(y_all[1:-1:2])
        
    
    h1_all=np.array([x_all[2*i]-x_all[2*i-2] for i in range(1,Nx+1,1)])
    q1_all=np.array([y_all[2*j]-y_all[2*j-2] for j in range(1,Ny+1,1)])
    
    h2_all=np.array([x_all[2*i+1]-x_all[2*i-1] for i in range(1,Nx,1)]+[x_all[-1]-x_all[-2]])
    q2_all=np.array([y_all[2*j+1]-y_all[2*j-1] for j in range(1,Ny,1)]+[y_all[-1]-y_all[-2]])
    
    h3_all=np.array([x_all[1]-x_all[0]]+[x_all[2*i-1]-x_all[2*i-3] for i in range(2,Nx+1,1)])
    q3_all=np.array([y_all[1]-y_all[0]]+[y_all[2*j-1]-y_all[2*j-3] for j in range(2,Ny+1,1)])
    
    h1=0.0
    h2=0.0
    h3=0.0
    
    q1=0.0
    q2=0.0
    q3=0.0
    
    Daa=0.0
    Dab=0.0
    Dbb=0.0
    Dba=0.0
    
    Daa1=0.0
    Dab1=0.0
    Dbb1=0.0
    Dba1=0.0
    
    Daa_1=0.0
    Dab_1=0.0
    Dbb_1=0.0
    Dba_1=0.0
    
    ind1_S1=(0,0)
    ind2_S1=(0,0)
    ind21_S1=(0,0)
    ind4_S1=(0,0)
    ind41_S1=(0,0)
    
    ind1_S2=(0,0)
    ind2_S2=(0,0)
    ind21_S2=(0,0)
    ind4_S2=(0,0)
    ind41_S2=(0,0)
    
    ind1_S3=(0,0)
    ind2_S3=(0,0)
    ind21_S3=(0,0)
    ind4_S3=(0,0)
    ind41_S3=(0,0)
    
    ind1_S4=(0,0)
    ind2_S4=(0,0)
    ind21_S4=(0,0)
    ind4_S4=(0,0)
    ind41_S4=(0,0)
    
    term_S1=(0.0,0.0,0.0,0.0)
    term_S2=(0.0,0.0,0.0,0.0)
    term_S3=(0.0,0.0,0.0,0.0)
    term_S4=(0.0,0.0,0.0,0.0)
    
    A=0.0
    B=0.0
    C=0.0
    D=0.0
    
    for j in range(1,Ny+1,1):
        for i in range(1,Nx+1,1):
            
            h1=h1_all[i-1]
            h2=h2_all[i-1]
            h3=h3_all[i-1]
            
            q1=q1_all[j-1]
            q2=q2_all[j-1]
            q3=q3_all[j-1]
            
            Daa=Dxx[2*j-1,2*i-1]
            Dab=Dyx[2*j-1,2*i-1]
            Dbb=Dyy[2*j-1,2*i-1]
            Dba=Dyx[2*j-1,2*i-1]
            
            if i==Nx:
                ind1_S1=(2*j-1,2*i)
                Daa1=Dxx[2*j-1,2*i]
                Dab1=Dyx[2*j-1,2*i]
            else:
                ind1_S1=(2*j-1,2*i+1)
                Daa1=Dxx[2*j-1,2*i+1]
                Dab1=Dyx[2*j-1,2*i+1]
                
            if j==Ny:
                ind2_S1=(2*j,2*i-1)
            else:
                ind2_S1=(2*j+1,2*i-1)
                
            if j==1:
                ind21_S1=(2*j-2,2*i-1)
            else:
                ind21_S1=(2*j-3,2*i-1)
            
            if j==1:
                if i==Nx:
                    ind4_S1=(2*j-2,2*i)
                else:
                    ind4_S1=(2*j-2,2*i+1)
            elif i==Nx:
                ind4_S1=(2*j-3,2*i)
            else:
                ind4_S1=(2*j-3,2*i+1)
            
            if j==Ny:
                if i==Nx:
                    ind41_S1=(2*j,2*i)
                else:
                    ind41_S1=(2*j,2*i+1)
            elif i==Nx:
                ind41_S1=(2*j+1,2*i)
            else:
                ind41_S1=(2*j+1,2*i+1)
                  
            if j==Ny:
                ind1_S2=(2*j,2*i-1)
                Dbb1=Dyy[2*j,2*i-1]
                Dba1=Dyx[2*j,2*i-1]
            else:
                ind1_S2=(2*j+1,2*i-1)
                Dbb1=Dyy[2*j+1,2*i-1]
                Dba1=Dyx[2*j+1,2*i-1]
             
            if i==Nx:
                ind2_S2=(2*j-1,2*i)
            else:
                ind2_S2=(2*j-1,2*i+1)
            
            if i==1:
                ind21_S2=(2*j-1,2*i-2)
            else:
                ind21_S2=(2*j-1,2*i-3)
            
            if i==1:
                if j==Ny:
                    ind4_S2=(2*j,2*i-2)
                else:
                    ind4_S2=(2*j+1,2*i-2)
            elif j==Ny:
                ind4_S2=(2*j,2*i-3)
            else:         
                ind4_S2=(2*j+1,2*i-3)
                
            if i==Nx:
                if j==Ny:
                    ind41_S2=(2*j,2*i)
                else:
                    ind41_S2=(2*j+1,2*i)
            elif j==Ny:
                ind41_S2=(2*j,2*i+1)
            else:
                ind41_S2=(2*j+1,2*i+1)
                      
                      
            if i==1:
                ind1_S3=(2*j-1,2*i-2)
                Daa_1=Dxx[2*j-1,2*i-2]
                Dab_1=Dyx[2*j-1,2*i-2]
            else:
                ind1_S3=(2*j-1,2*i-3)
                Daa_1=Dxx[2*j-1,2*i-3]
                Dab_1=Dyx[2*j-1,2*i-3]
            
            if j==1:
                ind2_S3=(2*j-2,2*i-1)
            else:
                ind2_S3=(2*j-3,2*i-1)
            
            if j==Ny:
                ind21_S3=(2*j,2*i-1)
            else:
                ind21_S3=(2*j+1,2*i-1)
            
            if j==Ny:
                if i==1:
                    ind4_S3=(2*j,2*i-2)
                else:
                    ind4_S3=(2*j,2*i-3)
            elif i==1:
                ind4_S3=(2*j+1,2*i-2)
            else:
                ind4_S3=(2*j+1,2*i-3)
            
            if j==1:
                if i==1:
                    ind41_S3=(2*j-2,2*i-2)
                else:
                    ind41_S3=(2*j-2,2*i-3)
            elif i==1:
                ind41_S3=(2*j-3,2*i-2)
            else:
                ind41_S3=(2*j-3,2*i-3)
                
            
            if j==1:
                ind1_S4=(2*j-2,2*i-1)
                Dbb_1=Dyy[2*j-2,2*i-1]
                Dba_1=Dyx[2*j-2,2*i-1]
            else:
                ind1_S4=(2*j-3,2*i-1)
                Dbb_1=Dyy[2*j-3,2*i-1]
                Dba_1=Dyx[2*j-3,2*i-1]
                
            if i==1:
                ind2_S4=(2*j-1,2*i-2)
            else:
                ind2_S4=(2*j-1,2*i-3)
            
            if i==Nx:
                ind21_S4=(2*j-1,2*i)
            else:
                ind21_S4=(2*j-1,2*i+1)
            
            if i==Nx:
                if j==1:
                    ind4_S4=(2*j-2,2*i)
                else:
                    ind4_S4=(2*j-3,2*i)
            elif j==1:
                ind4_S4=(2*j-2,2*i+1)
            else:
                ind4_S4=(2*j-3,2*i+1)
    
            if i==1:
                if j==1:
                    ind41_S4=(2*j-2,2*i-2)
                else:
                    ind41_S4=(2*j-3,2*i-2)
            elif j==1:
                ind41_S4=(2*j-2,2*i-3)
            else:
                ind41_S4=(2*j-3,2*i-3)

            A=0.0
            B=0.0
            C=0.0
            D=0.0
            
            term_S1=term_RNLMPFA(Daa,Daa1,Dab,Dab1,h2,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1,als,bes)
            term_S2=term_RNLMPFA(Dbb,Dbb1,Dba,Dba1,q2,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2,als,bes)
            term_S3=term_RNLMPFA(Daa,Daa_1,Dab,Dab_1,h3,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3,als,bes)
            term_S4=term_RNLMPFA(Dbb,Dbb_1,Dba,Dba_1,q3,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4,als,bes)

            A=q1*term_S1[0]
            B=q1*term_S3[0]
            C=h1*term_S2[0]
            D=h1*term_S4[0]
            
            diag0.append(A+B+C+D)
            
            if i<Nx:
                
                A=0.0
                C=0.0
                D=0.0
                
                A=q1*term_S1[1]
    
                C=h1*term_S2[2]
                
                D=h1*term_S4[3]
    
                diag1.append(A+C+D)

            if i>1:
                
                B=0.0
                C=0.0
                D=0.0
                
                B=q1*term_S3[1]
                          
                C=h1*term_S2[3]
                    
                D=h1*term_S4[2]
                
                diag_1.append(B+C+D)
                
            if j<Ny:
                
                A=0.0
                B=0.0
                C=0.0
                
                B=q1*term_S3[3]
                
                A=q1*term_S1[2]
    
                C=h1*term_S2[1]
    
                diag10.append(A+B+C)
                
                if i<Nx:
                    
                    A=0.0
                    C=0.0
                    
                    A=q1*term_S1[5]
                    
                    C=h1*term_S2[5]
                    
                    diag11.append(A+C)
                    
                if i>1:
                    
                    B=0.0
                    C=0.0
                    
                    B=q1*term_S3[4]
                        
                    C=h1*term_S2[4]
                        
                    diag1_1.append(B+C)
                    
            if j>1:
                
                A=0.0
                B=0.0
                D=0.0
                
                B=q1*term_S3[2]
                    
                A=q1*term_S1[3]
                    
                D=h1*term_S4[1]
                
                diag_10.append(A+B+D)
                
                if i<Nx:
                    
                    A=0.0
                    D=0.0
                    
                    A=q1*term_S1[4]
                    
                    D=h1*term_S4[4]
                    
                    diag_11.append(A+D)
                    
                if i>1:
                    
                    B=0.0
                    D=0.0
                    
                    B=q1*term_S3[5]
        
                    D=h1*term_S4[5]
                    
                    diag_1_1.append(B+D)
                            
        if j<Ny:
            diag1.append(0.0)
            diag_1.append(0.0)
            diag1_1.append(0.0)
            
        if j<Ny-1:
            diag11.append(0.0)
        
        if j<Ny and j>1:
            diag_1_1.append(0.0)
            
        if j>1:
            diag_11.append(0.0)
        
    diago0=np.array(diag0)
    diago1=np.array([0]+diag1)
    diago_1=np.array(diag_1+[0])
    
    diago10=np.array([0]*(Nx)+diag10)
    diago11=np.array([0]*(Nx+1)+diag11)
    diago1_1=np.array([0]*(Nx-1)+diag1_1)
    
    diago_10=np.array(diag_10+[0]*Nx)
    diago_11=np.array(diag_11+[0]*(Nx-1))
    diago_1_1=np.array(diag_1_1+[0]*(Nx+1))
    
    return((diago_1_1,diago_10,diago_11,diago_1,diago0,diago1,diago1_1,diago10,diago11))


def BC_RNLMPFA(Dxx,Dyy,Dyx,x_all,y_all,fs,U,als,bes):
    '''
    Generate RNLMPFA scheme BC vector
    '''

    mur_b=fs(x_all,y_all[0])    
    mur_d=fs(x_all[-1],y_all)   
    mur_h=fs(x_all,y_all[-1])   
    mur_g=fs(x_all[0],y_all)
        
    #Coin NW
    mur_h[0]=0.5*mur_h[0]
    mur_g[-1]=0.5*mur_g[-1]
    #Coin SE
    mur_b[-1]=0.5*mur_b[-1]
    mur_d[0]=0.5*mur_d[0]
    #Coin NE
    mur_d[-1]=0.5*mur_d[-1]
    mur_h[-1]=0.5*mur_h[-1]
    #Coin SW
    mur_b[0]=0.5*mur_b[0]
    mur_g[0]=0.5*mur_g[0]


    s=0
    
    Nx=len(x_all[1:-1:2])
    Ny=len(y_all[1:-1:2])
    
    S1=np.zeros(Nx*Ny)
    
    h1=0.0
    h2=0.0
    h3=0.0
    
    q1=0.0
    q2=0.0
    q3=0.0
    
    Daa=0.0
    Dab=0.0
    Dbb=0.0
    Dba=0.0
    
    Daa1=0.0
    Dab1=0.0
    Dbb1=0.0
    Dba1=0.0
    
    Daa_1=0.0
    Dab_1=0.0
    Dbb_1=0.0
    Dba_1=0.0
    
    ind1_S1=(0,0)
    ind2_S1=(0,0)
    ind21_S1=(0,0)
    ind4_S1=(0,0)
    ind41_S1=(0,0)
    
    ind1_S2=(0,0)
    ind2_S2=(0,0)
    ind21_S2=(0,0)
    ind4_S2=(0,0)
    ind41_S2=(0,0)
    
    ind1_S3=(0,0)
    ind2_S3=(0,0)
    ind21_S3=(0,0)
    ind4_S3=(0,0)
    ind41_S3=(0,0)
    
    ind1_S4=(0,0)
    ind2_S4=(0,0)
    ind21_S4=(0,0)
    ind4_S4=(0,0)
    ind41_S4=(0,0)
        
    A=0.0
    B=0.0
    C=0.0
    D=0.0
    
    term_S1=(0.0,0.0,0.0,0.0)
    term_S2=(0.0,0.0,0.0,0.0)
    term_S3=(0.0,0.0,0.0,0.0)
    term_S4=(0.0,0.0,0.0,0.0)
    
    for j in range(1,Ny+1,1):
        for i in range(1,Nx+1,1):
            
            if i==1:
                
                h1=x_all[2*i+1-1]-x_all[2*i-1-1]# xi+1/2 - xi-1/2  
                q1=y_all[2*j+1-1]-y_all[2*j-1-1]# yi+1/2 - yi-1/2
                
                h3=x_all[2*i-1]-x_all[2*i-2]
                h2=x_all[2*i+2-1]-x_all[2*i-1]
    
                if j==Ny:
                    q2=y_all[2*j]-y_all[2*j-1]
                else:
                    q2=y_all[2*j+2-1]-y_all[2*j-1]
                    
                if j==1:
                    q3=y_all[2*j-1]-y_all[2*j-2]
                else:
                    q3=y_all[2*j-1]-y_all[2*j-2-1]
                    
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
            
                if j==Ny:
                    ind1_S2=(2*j,2*i-1)
                    Dbb1=Dyy[2*j,2*i-1]
                    Dba1=Dyx[2*j,2*i-1]
                else:
                    ind1_S2=(2*j+1,2*i-1)
                    Dbb1=Dyy[2*j+1,2*i-1]
                    Dba1=Dyx[2*j+1,2*i-1]
                 
                if i==Nx:
                    ind2_S2=(2*j-1,2*i)
                else:
                    ind2_S2=(2*j-1,2*i+1)
                
                if i==1:
                    ind21_S2=(2*j-1,2*i-2)
                else:
                    ind21_S2=(2*j-1,2*i-3)
                
                if i==1:
                    if j==Ny:
                        ind4_S2=(2*j,2*i-2)
                    else:
                        ind4_S2=(2*j+1,2*i-2)
                elif j==Ny:
                    ind4_S2=(2*j,2*i-3)
                else:         
                    ind4_S2=(2*j+1,2*i-3)
                    
                if i==Nx:
                    if j==Ny:
                        ind41_S2=(2*j,2*i)
                    else:
                        ind41_S2=(2*j+1,2*i)
                elif j==Ny:
                    ind41_S2=(2*j,2*i+1)
                else:
                    ind41_S2=(2*j+1,2*i+1)
                          
                          
                if i==1:
                    ind1_S3=(2*j-1,2*i-2)
                    Daa_1=Dxx[2*j-1,2*i-2]
                    Dab_1=Dyx[2*j-1,2*i-2]
                else:
                    ind1_S3=(2*j-1,2*i-3)
                    Daa_1=Dxx[2*j-1,2*i-3]
                    Dab_1=Dyx[2*j-1,2*i-3]
                
                if j==1:
                    ind2_S3=(2*j-2,2*i-1)
                else:
                    ind2_S3=(2*j-3,2*i-1)
                
                if j==Ny:
                    ind21_S3=(2*j,2*i-1)
                else:
                    ind21_S3=(2*j+1,2*i-1)
                
                if j==Ny:
                    if i==1:
                        ind4_S3=(2*j,2*i-2)
                    else:
                        ind4_S3=(2*j,2*i-3)
                elif i==1:
                    ind4_S3=(2*j+1,2*i-2)
                else:
                    ind4_S3=(2*j+1,2*i-3)
                
                if j==1:
                    if i==1:
                        ind41_S3=(2*j-2,2*i-2)
                    else:
                        ind41_S3=(2*j-2,2*i-3)
                elif i==1:
                    ind41_S3=(2*j-3,2*i-2)
                else:
                    ind41_S3=(2*j-3,2*i-3)
                    
                if j==1:
                    ind1_S4=(2*j-2,2*i-1)
                    Dbb_1=Dyy[2*j-2,2*i-1]
                    Dba_1=Dyx[2*j-2,2*i-1]
                else:
                    ind1_S4=(2*j-3,2*i-1)
                    Dbb_1=Dyy[2*j-3,2*i-1]
                    Dba_1=Dyx[2*j-3,2*i-1]
                    
                if i==1:
                    ind2_S4=(2*j-1,2*i-2)
                else:
                    ind2_S4=(2*j-1,2*i-3)
                
                if i==Nx:
                    ind21_S4=(2*j-1,2*i)
                else:
                    ind21_S4=(2*j-1,2*i+1)
                
                if i==Nx:
                    if j==1:
                        ind4_S4=(2*j-2,2*i)
                    else:
                        ind4_S4=(2*j-3,2*i)
                elif j==1:
                    ind4_S4=(2*j-2,2*i+1)
                else:
                    ind4_S4=(2*j-3,2*i+1)
        
                if i==1:
                    if j==1:
                        ind41_S4=(2*j-2,2*i-2)
                    else:
                        ind41_S4=(2*j-3,2*i-2)
                elif j==1:
                    ind41_S4=(2*j-2,2*i-3)
                else:
                    ind41_S4=(2*j-3,2*i-3)

                B=0.0
                C=0.0
                D=0.0
                
                E=0.0
                F=0.0
                
                G=0.0
                H=0.0
                
                term_S2=term_RNLMPFA(Dbb,Dbb1,Dba,Dba1,q2,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2,als,bes)
                term_S3=term_RNLMPFA(Daa,Daa_1,Dab,Dab_1,h3,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3,als,bes)
                term_S4=term_RNLMPFA(Dbb,Dbb_1,Dba,Dba_1,q3,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4,als,bes)
                
                B=q1*term_S3[1]
                C=h1*term_S2[3]
                D=h1*term_S4[2]

                E=q1*term_S3[5]
                F=h1*term_S4[5]

                G=q1*term_S3[4]
                H=h1*term_S2[4]

                if j==1:
                    a=mur_g[2*j-2]
                else:
                    a=mur_g[2*j-3]
                if j==Ny:
                    b=mur_g[2*j]
                else:
                    b=mur_g[2*j+1]

                S1[s]=S1[s]+mur_g[2*j-1]*(B+C+D)+a*(F)+b*(H)+mur_g[2*j-2]*(E)+mur_g[2*j]*(G)
            
            if j==1:
                
                h1=x_all[2*i+1-1]-x_all[2*i-1-1]# xi+1/2 - xi-1/2  
                q1=y_all[2*j+1-1]-y_all[2*j-1-1]# yi+1/2 - yi-1/2
                
                if i==1:
                    h3=x_all[2*i-1]-x_all[2*i-2]
                else:
                    h3=x_all[2*i-1]-x_all[2*i-2-1]
                    
                if i==Nx:
                    h2=x_all[2*i]-x_all[2*i-1]
                else:
                    h2=x_all[2*i+2-1]-x_all[2*i-1]
    
                q2=y_all[2*j+2-1]-y_all[2*j-1]
                q3=y_all[2*j-1]-y_all[2*j-2]
                
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
            
                if i==Nx:
                    ind1_S1=(2*j-1,2*i)
                    Daa1=Dxx[2*j-1,2*i]
                    Dab1=Dyx[2*j-1,2*i]
                else:
                    ind1_S1=(2*j-1,2*i+1)
                    Daa1=Dxx[2*j-1,2*i+1]
                    Dab1=Dyx[2*j-1,2*i+1]
                    
                if j==Ny:
                    ind2_S1=(2*j,2*i-1)
                else:
                    ind2_S1=(2*j+1,2*i-1)
                    
                if j==1:
                    ind21_S1=(2*j-2,2*i-1)
                else:
                    ind21_S1=(2*j-3,2*i-1)
                
                if j==1:
                    if i==Nx:
                        ind4_S1=(2*j-2,2*i)
                    else:
                        ind4_S1=(2*j-2,2*i+1)
                elif i==Nx:
                    ind4_S1=(2*j-3,2*i)
                else:
                    ind4_S1=(2*j-3,2*i+1)
                
                if j==Ny:
                    if i==Nx:
                        ind41_S1=(2*j,2*i)
                    else:
                        ind41_S1=(2*j,2*i+1)
                elif i==Nx:
                    ind41_S1=(2*j+1,2*i)
                else:
                    ind41_S1=(2*j+1,2*i+1)
                      
                if i==1:
                    ind1_S3=(2*j-1,2*i-2)
                    Daa_1=Dxx[2*j-1,2*i-2]
                    Dab_1=Dyx[2*j-1,2*i-2]
                else:
                    ind1_S3=(2*j-1,2*i-3)
                    Daa_1=Dxx[2*j-1,2*i-3]
                    Dab_1=Dyx[2*j-1,2*i-3]
                
                if j==1:
                    ind2_S3=(2*j-2,2*i-1)
                else:
                    ind2_S3=(2*j-3,2*i-1)
                
                if j==Ny:
                    ind21_S3=(2*j,2*i-1)
                else:
                    ind21_S3=(2*j+1,2*i-1)
                
                if j==Ny:
                    if i==1:
                        ind4_S3=(2*j,2*i-2)
                    else:
                        ind4_S3=(2*j,2*i-3)
                elif i==1:
                    ind4_S3=(2*j+1,2*i-2)
                else:
                    ind4_S3=(2*j+1,2*i-3)
                
                if j==1:
                    if i==1:
                        ind41_S3=(2*j-2,2*i-2)
                    else:
                        ind41_S3=(2*j-2,2*i-3)
                elif i==1:
                    ind41_S3=(2*j-3,2*i-2)
                else:
                    ind41_S3=(2*j-3,2*i-3)
                    
                
                if j==1:
                    ind1_S4=(2*j-2,2*i-1)
                    Dbb_1=Dyy[2*j-2,2*i-1]
                    Dba_1=Dyx[2*j-2,2*i-1]
                else:
                    ind1_S4=(2*j-3,2*i-1)
                    Dbb_1=Dyy[2*j-3,2*i-1]
                    Dba_1=Dyx[2*j-3,2*i-1]
                    
                if i==1:
                    ind2_S4=(2*j-1,2*i-2)
                else:
                    ind2_S4=(2*j-1,2*i-3)
                
                if i==Nx:
                    ind21_S4=(2*j-1,2*i)
                else:
                    ind21_S4=(2*j-1,2*i+1)
                
                if i==Nx:
                    if j==1:
                        ind4_S4=(2*j-2,2*i)
                    else:
                        ind4_S4=(2*j-3,2*i)
                elif j==1:
                    ind4_S4=(2*j-2,2*i+1)
                else:
                    ind4_S4=(2*j-3,2*i+1)
        
                if i==1:
                    if j==1:
                        ind41_S4=(2*j-2,2*i-2)
                    else:
                        ind41_S4=(2*j-3,2*i-2)
                elif j==1:
                    ind41_S4=(2*j-2,2*i-3)
                else:
                    ind41_S4=(2*j-3,2*i-3)

                A=0.0
                B=0.0
                D=0.0
                
                E=0.0
                F=0.0
                
                G=0.0
                H=0.0
                
                term_S1=term_RNLMPFA(Daa,Daa1,Dab,Dab1,h2,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1,als,bes)
                term_S3=term_RNLMPFA(Daa,Daa_1,Dab,Dab_1,h3,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3,als,bes)
                term_S4=term_RNLMPFA(Dbb,Dbb_1,Dba,Dba_1,q3,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4,als,bes)
            
                B=q1*term_S3[2]
                A=q1*term_S1[3]
                D=h1*term_S4[1]
                
                G=q1*term_S3[5]
                H=h1*term_S4[5]

                E=q1*term_S1[4]
                F=h1*term_S4[4]

                if i==Nx:
                    a=mur_b[2*i]
                else:
                    a=mur_b[2*i+1]
                if i==1:
                    b=mur_b[2*i-2]
                else:
                    b=mur_b[2*i-3]


                S1[s]=S1[s] + mur_b[2*i-1]*(A+B+D)+a*(E)+b*(G)+mur_b[2*i]*(F)+mur_b[2*i-2]*(H)
                
            if i==Nx:
                
                h1=x_all[2*i+1-1]-x_all[2*i-1-1]# xi+1/2 - xi-1/2  
                q1=y_all[2*j+1-1]-y_all[2*j-1-1]# yi+1/2 - yi-1/2
                
                h3=x_all[2*i-1]-x_all[2*i-2-1]
                h2=x_all[2*i]-x_all[2*i-1]
               
                if j==Ny:
                    q2=y_all[2*j]-y_all[2*j-1]
                else:
                    q2=y_all[2*j+2-1]-y_all[2*j-1]
                
                if j==1:
                    q3=y_all[2*j-1]-y_all[2*j-2]
                else:
                    q3=y_all[2*j-1]-y_all[2*j-2-1]
                    
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
            
                if i==Nx:
                    ind1_S1=(2*j-1,2*i)
                    Daa1=Dxx[2*j-1,2*i]
                    Dab1=Dyx[2*j-1,2*i]
                else:
                    ind1_S1=(2*j-1,2*i+1)
                    Daa1=Dxx[2*j-1,2*i+1]
                    Dab1=Dyx[2*j-1,2*i+1]
                    
                if j==Ny:
                    ind2_S1=(2*j,2*i-1)
                else:
                    ind2_S1=(2*j+1,2*i-1)
                    
                if j==1:
                    ind21_S1=(2*j-2,2*i-1)
                else:
                    ind21_S1=(2*j-3,2*i-1)
                
                if j==1:
                    if i==Nx:
                        ind4_S1=(2*j-2,2*i)
                    else:
                        ind4_S1=(2*j-2,2*i+1)
                elif i==Nx:
                    ind4_S1=(2*j-3,2*i)
                else:
                    ind4_S1=(2*j-3,2*i+1)
                
                if j==Ny:
                    if i==Nx:
                        ind41_S1=(2*j,2*i)
                    else:
                        ind41_S1=(2*j,2*i+1)
                elif i==Nx:
                    ind41_S1=(2*j+1,2*i)
                else:
                    ind41_S1=(2*j+1,2*i+1)
                      
                if j==Ny:
                    ind1_S2=(2*j,2*i-1)
                    Dbb1=Dyy[2*j,2*i-1]
                    Dba1=Dyx[2*j,2*i-1]
                else:
                    ind1_S2=(2*j+1,2*i-1)
                    Dbb1=Dyy[2*j+1,2*i-1]
                    Dba1=Dyx[2*j+1,2*i-1]
                 
                if i==Nx:
                    ind2_S2=(2*j-1,2*i)
                else:
                    ind2_S2=(2*j-1,2*i+1)
                
                if i==1:
                    ind21_S2=(2*j-1,2*i-2)
                else:
                    ind21_S2=(2*j-1,2*i-3)
                
                if i==1:
                    if j==Ny:
                        ind4_S2=(2*j,2*i-2)
                    else:
                        ind4_S2=(2*j+1,2*i-2)
                elif j==Ny:
                    ind4_S2=(2*j,2*i-3)
                else:         
                    ind4_S2=(2*j+1,2*i-3)
                    
                if i==Nx:
                    if j==Ny:
                        ind41_S2=(2*j,2*i)
                    else:
                        ind41_S2=(2*j+1,2*i)
                elif j==Ny:
                    ind41_S2=(2*j,2*i+1)
                else:
                    ind41_S2=(2*j+1,2*i+1)
                                          
                if j==1:
                    ind1_S4=(2*j-2,2*i-1)
                    Dbb_1=Dyy[2*j-2,2*i-1]
                    Dba_1=Dyx[2*j-2,2*i-1]
                else:
                    ind1_S4=(2*j-3,2*i-1)
                    Dbb_1=Dyy[2*j-3,2*i-1]
                    Dba_1=Dyx[2*j-3,2*i-1]
                    
                if i==1:
                    ind2_S4=(2*j-1,2*i-2)
                else:
                    ind2_S4=(2*j-1,2*i-3)
                
                if i==Nx:
                    ind21_S4=(2*j-1,2*i)
                else:
                    ind21_S4=(2*j-1,2*i+1)
                
                if i==Nx:
                    if j==1:
                        ind4_S4=(2*j-2,2*i)
                    else:
                        ind4_S4=(2*j-3,2*i)
                elif j==1:
                    ind4_S4=(2*j-2,2*i+1)
                else:
                    ind4_S4=(2*j-3,2*i+1)
        
                if i==1:
                    if j==1:
                        ind41_S4=(2*j-2,2*i-2)
                    else:
                        ind41_S4=(2*j-3,2*i-2)
                elif j==1:
                    ind41_S4=(2*j-2,2*i-3)
                else:
                    ind41_S4=(2*j-3,2*i-3)

                A=0.0
                C=0.0
                D=0.0
                
                E=0.0
                F=0.0
                
                G=0.0
                H=0.0
                
                term_S1=term_RNLMPFA(Daa,Daa1,Dab,Dab1,h2,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1,als,bes)
                term_S2=term_RNLMPFA(Dbb,Dbb1,Dba,Dba1,q2,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2,als,bes)
                term_S4=term_RNLMPFA(Dbb,Dbb_1,Dba,Dba_1,q3,h3,h2,Nx,Ny,i,j,'S4',U,ind1_S4,ind2_S4,ind21_S4,ind4_S4,ind41_S4,als,bes)
    
                A=q1*term_S1[1]
                C=h1*term_S2[2]
                D=h1*term_S4[3]
                
                E=q1*term_S1[5]
                F=h1*term_S2[5]
                                   
                G=q1*term_S1[4]
                H=h1*term_S4[4]
                
                if j==Ny:
                    a=mur_d[2*j]
                else:
                    a=mur_d[2*j+1]
                if j==1:
                    b=mur_d[2*j-2]
                else:
                    b=mur_d[2*j-3]

                S1[s]=S1[s] + mur_d[2*j-1]*(A+C+D)+a*(F)+b*(H)+mur_d[2*j]*(E)+mur_d[2*j-2]*(G)
                
                
                
            if j==Ny:
                
                h1=x_all[2*i+1-1]-x_all[2*i-1-1]# xi+1/2 - xi-1/2  
                q1=y_all[2*j+1-1]-y_all[2*j-1-1]# yi+1/2 - yi-1/2
                
                if i==1:
                    h3=x_all[2*i-1]-x_all[2*i-2]
                else:
                    h3=x_all[2*i-1]-x_all[2*i-2-1]
                    
                if i==Nx:
                    h2=x_all[2*i]-x_all[2*i-1]
                else:
                    h2=x_all[2*i+2-1]-x_all[2*i-1]
               
                q2=y_all[2*j]-y_all[2*j-1]
                q3=y_all[2*j-1]-y_all[2*j-2-1]
                                
                Daa=Dxx[2*j-1,2*i-1]
                Dab=Dyx[2*j-1,2*i-1]
                Dbb=Dyy[2*j-1,2*i-1]
                Dba=Dyx[2*j-1,2*i-1]
            
                if i==Nx:
                    ind1_S1=(2*j-1,2*i)
                    Daa1=Dxx[2*j-1,2*i]
                    Dab1=Dyx[2*j-1,2*i]
                else:
                    ind1_S1=(2*j-1,2*i+1)
                    Daa1=Dxx[2*j-1,2*i+1]
                    Dab1=Dyx[2*j-1,2*i+1]
                    
                if j==Ny:
                    ind2_S1=(2*j,2*i-1)
                else:
                    ind2_S1=(2*j+1,2*i-1)
                    
                if j==1:
                    ind21_S1=(2*j-2,2*i-1)
                else:
                    ind21_S1=(2*j-3,2*i-1)
                
                if j==1:
                    if i==Nx:
                        ind4_S1=(2*j-2,2*i)
                    else:
                        ind4_S1=(2*j-2,2*i+1)
                elif i==Nx:
                    ind4_S1=(2*j-3,2*i)
                else:
                    ind4_S1=(2*j-3,2*i+1)
                
                if j==Ny:
                    if i==Nx:
                        ind41_S1=(2*j,2*i)
                    else:
                        ind41_S1=(2*j,2*i+1)
                elif i==Nx:
                    ind41_S1=(2*j+1,2*i)
                else:
                    ind41_S1=(2*j+1,2*i+1)
                      
                if j==Ny:
                    ind1_S2=(2*j,2*i-1)
                    Dbb1=Dyy[2*j,2*i-1]
                    Dba1=Dyx[2*j,2*i-1]
                else:
                    ind1_S2=(2*j+1,2*i-1)
                    Dbb1=Dyy[2*j+1,2*i-1]
                    Dba1=Dyx[2*j+1,2*i-1]
                 
                if i==Nx:
                    ind2_S2=(2*j-1,2*i)
                else:
                    ind2_S2=(2*j-1,2*i+1)
                
                if i==1:
                    ind21_S2=(2*j-1,2*i-2)
                else:
                    ind21_S2=(2*j-1,2*i-3)
                
                if i==1:
                    if j==Ny:
                        ind4_S2=(2*j,2*i-2)
                    else:
                        ind4_S2=(2*j+1,2*i-2)
                elif j==Ny:
                    ind4_S2=(2*j,2*i-3)
                else:         
                    ind4_S2=(2*j+1,2*i-3)
                    
                if i==Nx:
                    if j==Ny:
                        ind41_S2=(2*j,2*i)
                    else:
                        ind41_S2=(2*j+1,2*i)
                elif j==Ny:
                    ind41_S2=(2*j,2*i+1)
                else:
                    ind41_S2=(2*j+1,2*i+1)
                          
                          
                if i==1:
                    ind1_S3=(2*j-1,2*i-2)
                    Daa_1=Dxx[2*j-1,2*i-2]
                    Dab_1=Dyx[2*j-1,2*i-2]
                else:
                    ind1_S3=(2*j-1,2*i-3)
                    Daa_1=Dxx[2*j-1,2*i-3]
                    Dab_1=Dyx[2*j-1,2*i-3]
                
                if j==1:
                    ind2_S3=(2*j-2,2*i-1)
                else:
                    ind2_S3=(2*j-3,2*i-1)
                
                if j==Ny:
                    ind21_S3=(2*j,2*i-1)
                else:
                    ind21_S3=(2*j+1,2*i-1)
                
                if j==Ny:
                    if i==1:
                        ind4_S3=(2*j,2*i-2)
                    else:
                        ind4_S3=(2*j,2*i-3)
                elif i==1:
                    ind4_S3=(2*j+1,2*i-2)
                else:
                    ind4_S3=(2*j+1,2*i-3)
                
                if j==1:
                    if i==1:
                        ind41_S3=(2*j-2,2*i-2)
                    else:
                        ind41_S3=(2*j-2,2*i-3)
                elif i==1:
                    ind41_S3=(2*j-3,2*i-2)
                else:
                    ind41_S3=(2*j-3,2*i-3)
                    
                
                A=0.0
                B=0.0
                C=0.0
                D=0.0
                
                E=0.0
                F=0.0
                
                G=0.0
                H=0.0
                
                term_S1=term_RNLMPFA(Daa,Daa1,Dab,Dab1,h2,q2,q3,Nx,Ny,i,j,'S1',U,ind1_S1,ind2_S1,ind21_S1,ind4_S1,ind41_S1,als,bes)
                term_S2=term_RNLMPFA(Dbb,Dbb1,Dba,Dba1,q2,h2,h3,Nx,Ny,i,j,'S2',U,ind1_S2,ind2_S2,ind21_S2,ind4_S2,ind41_S2,als,bes)
                term_S3=term_RNLMPFA(Daa,Daa_1,Dab,Dab_1,h3,q3,q2,Nx,Ny,i,j,'S3',U,ind1_S3,ind2_S3,ind21_S3,ind4_S3,ind41_S3,als,bes)

                B=q1*term_S3[3]
                A=q1*term_S1[2]
                C=h1*term_S2[1]
                
                E=q1*term_S1[5]
                F=h1*term_S2[5]
                
                G=q1*term_S3[4]
                H=h1*term_S2[4]
                
                if i==Nx:
                    a=mur_h[2*i]
                else:
                    a=mur_h[2*i+1]
                
                if i==1:
                    b=mur_h[2*i-2]
                else:
                    b=mur_h[2*i-3]
                
                S1[s]=S1[s] + mur_h[2*i-1]*(A+B+C)+a*(E)+b*(G)+mur_h[2*i]*(F)+mur_h[2*i-2]*(H)
                
            s=s+1
            
    return(S1)

    
def mapping(U1,fs,x_all,y_all,x_cell,y_cell):
    '''
      map the nonlinear solution over the whole FV mesh
    '''
    mur_b=fs(x_all,y_all[0])    
    mur_d=fs(x_all[-1],y_all)   
    mur_h=fs(x_all,y_all[-1])   
    mur_g=fs(x_all[0],y_all)
    
    f_interp=interp2d(x_cell,y_cell,U1,"linear")
    f_all=abs(f_interp(x_all[1:-1],y_all[1:-1]))
    U=np.zeros((len(y_all),len(x_all)))
    
    U[1:-1,1:-1]=f_all
    U[0]=mur_b
    U[-1]=mur_h
    U[:,0]=mur_g
    U[:,-1]=mur_d
                
    return(abs(U))

@numba.jit(nopython=True)
def sgn(x):
    '''
    equal to 1 is x>0 else equal to 0
    '''
    y=0.0
    
    if x>0.0:
        y=1.0
    else:
        y=0.0
    
    return(y)
        
@numba.jit(nopython=True)
def calc_nu(a1,a2):
    '''
    nonlinear weights calculation
    '''
    nu1=0.0
    nu2=0.0
    
    if a1*a2<0.0:
        print('PROBLEM a1 a2')
    if a1+a2==0.0:
        nu1=0.5
        nu2=0.5
    else:
        nu1=(a2)/(a1+a2)
        nu2=(a1)/(a1+a2)
    return(nu1,nu2)

def c1c2(x_all,y_all,Dxx,Dyy,Dyx):
    '''
    c1 c2 computation for the R-NLMPFA scheme
    '''
    
    Nx=len(x_all[1:-1:2])
    Ny=len(y_all[1:-1:2])
    
    lambda1_sigma1=np.zeros((Ny,Nx))
    lambda1_sigma2=np.zeros((Ny,Nx))
    lambda1_sigma3=np.zeros((Ny,Nx))
    lambda1_sigma4=np.zeros((Ny,Nx))
    
    lambda2_sigma1=np.zeros((Ny,Nx))
    lambda2_sigma2=np.zeros((Ny,Nx))
    lambda2_sigma3=np.zeros((Ny,Nx))
    lambda2_sigma4=np.zeros((Ny,Nx))
    
    nu1_sigma1=np.zeros((Ny,Nx))
    nu1_sigma2=np.zeros((Ny,Nx))
    nu1_sigma3=np.zeros((Ny,Nx))
    nu1_sigma4=np.zeros((Ny,Nx))
    
    nu2_sigma1=np.zeros((Ny,Nx))
    nu2_sigma2=np.zeros((Ny,Nx))
    nu2_sigma3=np.zeros((Ny,Nx))
    nu2_sigma4=np.zeros((Ny,Nx))
    
    for j in range(1,Ny+1,1):
        for i in range(1,Nx+1,1):
            
            sigma1=abs(y_all[2*j]-y_all[2*j-2])
            sigma3=abs(y_all[2*j]-y_all[2*j-2])
            sigma2=abs(x_all[2*i]-x_all[2*i-2])
            sigma4=abs(x_all[2*i]-x_all[2*i-2])
            
            Dxx_K=Dxx[2*j-1,2*i-1]
            Dyy_K=Dyy[2*j-1,2*i-1]
            Dyx_K=Dyx[2*j-1,2*i-1]
            
            if i==Nx:
                Dxx_L_sigma1=Dxx[2*j-1,2*i]
            else:
                Dxx_L_sigma1=Dxx[2*j-1,2*i+1]
            if i==1:
                Dxx_L_sigma3=Dxx[2*j-1,2*i-2]
            else:
                Dxx_L_sigma3=Dxx[2*j-1,2*i-3]
            if j==Ny:
                Dyy_L_sigma2=Dyy[2*j,2*i-1]
            else:
                Dyy_L_sigma2=Dyy[2*j+1,2*i-1]
            if j==1:
                Dyy_L_sigma4=Dyy[2*j-2,2*i-1]
            else:
                Dyy_L_sigma4=Dyy[2*j-3,2*i-1]
            
            if i==Nx:
                Dyx_L_sigma1=Dyx[2*j-1,2*i]
            else:
                Dyx_L_sigma1=Dyx[2*j-1,2*i+1]
            if i==1:
                Dyx_L_sigma3=Dyx[2*j-1,2*i-2]
            else:
                Dyx_L_sigma3=Dyx[2*j-1,2*i-3]
            if j==Ny:
                Dyx_L_sigma2=Dyx[2*j,2*i-1]
            else:
                Dyx_L_sigma2=Dyx[2*j+1,2*i-1]
            if j==1:
                Dyx_L_sigma4=Dyx[2*j-2,2*i-1]
            else:
                Dyx_L_sigma4=Dyx[2*j-3,2*i-1]

            if i==Nx:
                dx_sigma1=abs(x_all[2*i]-x_all[2*i-1])
            else:
                dx_sigma1=abs(x_all[2*i+1]-x_all[2*i-1])
            if i==1:
                dx_sigma3=abs(x_all[2*i-1]-x_all[2*i-2])
            else:
                dx_sigma3=abs(x_all[2*i-1]-x_all[2*i-3])
            if j==Ny:
                dy_sigma2=abs(y_all[2*j]-y_all[2*j-1])
            else:
                dy_sigma2=abs(y_all[2*j+1]-y_all[2*j-1])
            if j==1:
                dy_sigma4=abs(y_all[2*j-1]-y_all[2*j-2])
            else:
                dy_sigma4=abs(y_all[2*j-1]-y_all[2*j-3])
            
            lambda1_sigma1[j-1,i-1]=sigma1*Dxx_K/dx_sigma1
            lambda2_sigma1[j-1,i-1]=sigma1*Dxx_L_sigma1/dx_sigma1
            lambda1_sigma3[j-1,i-1]=sigma3*Dxx_K/dx_sigma3
            lambda2_sigma3[j-1,i-1]=sigma3*Dxx_L_sigma3/dx_sigma3
            
            lambda1_sigma2[j-1,i-1]=sigma2*Dyy_K/dy_sigma2
            lambda2_sigma2[j-1,i-1]=sigma2*Dyy_L_sigma2/dy_sigma2
            lambda1_sigma4[j-1,i-1]=sigma4*Dyy_K/dy_sigma4
            lambda2_sigma4[j-1,i-1]=sigma4*Dyy_L_sigma4/dy_sigma4
            
            nu1_sigma1[j-1,i-1]=sigma1*(sgn(Dyx_K)*(Dyx_K)/dy_sigma2+sgn(-Dyx_K)*abs(Dyx_K)/dy_sigma4)
            nu1_sigma3[j-1,i-1]=sigma3*(sgn(Dyx_K)*(Dyx_K)/dy_sigma4+sgn(-Dyx_K)*abs(Dyx_K)/dy_sigma2)
            nu1_sigma2[j-1,i-1]=sigma2*(sgn(Dyx_K)*(Dyx_K)/dx_sigma1+sgn(-Dyx_K)*abs(Dyx_K)/dx_sigma3)
            nu1_sigma4[j-1,i-1]=sigma4*(sgn(Dyx_K)*(Dyx_K)/dx_sigma3+sgn(-Dyx_K)*abs(Dyx_K)/dx_sigma1)
            
            nu2_sigma1[j-1,i-1]=sigma1*(sgn(Dyx_L_sigma1)*(Dyx_L_sigma1)/dy_sigma4+sgn(-Dyx_L_sigma1)*abs(Dyx_L_sigma1)/dy_sigma2)
            nu2_sigma3[j-1,i-1]=sigma3*(sgn(Dyx_L_sigma3)*(Dyx_L_sigma3)/dy_sigma2+sgn(-Dyx_L_sigma3)*abs(Dyx_L_sigma3)/dy_sigma4)
            nu2_sigma2[j-1,i-1]=sigma2*(sgn(Dyx_L_sigma2)*(Dyx_L_sigma2)/dx_sigma3+sgn(-Dyx_L_sigma2)*abs(Dyx_L_sigma2)/dx_sigma1)
            nu2_sigma4[j-1,i-1]=sigma4*(sgn(Dyx_L_sigma4)*(Dyx_L_sigma4)/dx_sigma1+sgn(-Dyx_L_sigma4)*abs(Dyx_L_sigma4)/dx_sigma3)
            
    c=[] 
    for j in range(2,Ny,1):
        for i in range(2,Nx,1):
            A1=max(lambda1_sigma1[j-1,i-1],lambda2_sigma1[j-1,i-1])+max(lambda1_sigma2[j-1,i-1],lambda2_sigma2[j-1,i-1])+max(lambda1_sigma3[j-1,i-1],lambda2_sigma3[j-1,i-1])+max(lambda1_sigma4[j-1,i-1],lambda2_sigma4[j-1,i-1])+2*(nu1_sigma1[j-1,i-1]+nu1_sigma2[j-1,i-1]+nu1_sigma3[j-1,i-1]+nu1_sigma4[j-1,i-1])
            cond1=(min(lambda1_sigma2[j-1,i-1],lambda2_sigma2[j-1,i-1])+min(lambda1_sigma4[j-1,i-1],lambda2_sigma4[j-1,i-1]))/max(nu2_sigma1[j-1,i-1],nu2_sigma3[j-1,i-1])
            cond2=(min(lambda1_sigma1[j-1,i-1],lambda2_sigma1[j-1,i-1])*min(lambda1_sigma2[j-2,i-1],lambda2_sigma2[j-2,i-1]))/(max(nu2_sigma1[j-2,i-1],nu2_sigma2[j-2,i-1])*A1)
            cond3=(min(lambda1_sigma3[j-1,i-1],lambda2_sigma3[j-1,i-1])*min(lambda1_sigma2[j-2,i-1],lambda2_sigma2[j-2,i-1]))/(max(nu2_sigma2[j-2,i-1],nu2_sigma3[j-2,i-1])*A1)
            cond4=(min(lambda1_sigma3[j-1,i-1],lambda2_sigma3[j-1,i-1])*min(lambda1_sigma4[j,i-1],lambda2_sigma4[j,i-1]))/(max(nu2_sigma3[j,i-1],nu2_sigma4[j,i-1])*A1)
            cond5=(min(lambda1_sigma1[j-1,i-1],lambda2_sigma1[j-1,i-1])*min(lambda1_sigma4[j,i-1],lambda2_sigma4[j,i-1]))/(max(nu2_sigma1[j,i-1],nu2_sigma4[j,i-1])*A1)
            
            c.append(min(cond1,cond2,cond3,cond4,cond5)/15)
            
    return(c)         
        


def verif_cond_nordbotten(x_all,y_all,Mat):
    '''
    verify the validity of the nordbotten conditions
    '''
    Nx=len(x_all[1:-1:2])
    Ny=len(y_all[1:-1:2])
    
    k=0
    
    res=np.zeros((Ny,Nx))
    
    for j in range(1,Ny+1,1):
        for i in range(1,Nx+1,1):
            
            if i==1 or j==1 or i==Nx or j==Ny:
                
                pass
            
            else:
            
                row0=Mat[k]
                row1=Mat[k+Nx]
                row_1=Mat[k-Nx]
                
                m10=row0[k]
                m20=row0[k+1]
                m40=row0[k+Nx]
                m60=row0[k-1]
                m80=row0[k-Nx]
                
                m30=row0[k+Nx+1]
                m50=row0[k+Nx-1]
                m70=row0[k-Nx-1]
                m90=row0[k-Nx+1]
                
                m71=row1[k+Nx-Nx-1]
                m81=row1[k+Nx-Nx]
                m91=row1[k+Nx-Nx+1]
                
                m3_1=row_1[k-Nx+Nx+1]
                m4_1=row_1[k-Nx+Nx]
                m5_1=row_1[k-Nx+Nx-1]
                
                A0=m10>0.0
                A1a=m20<0.0
                A1c=m60<0.0
                A1b=m40<0.0
                A1d=m80<0.0
                
                A2=(m10+m20+m60)>0.0
            
                A3a=m20*m4_1-(m3_1*m10)>0.0
                A3b=m60*m4_1-(m5_1*m10)>0.0
                A3c=m20*m81-(m91*m10)>0.0
                A3d=m60*m81-(m71*m10)>0.0
                
                res[j-1,i-1]=max(A0,A1a,A1b,A1c,A1d,A2,A3a,A3b,A3c,A3d)*True
            
            k=k+1
            
            return(res.sum())


def Picard_algo_NLTPFA_NLMPFA(x_all,y_all,x_cell,y_cell,D1,D2,D12,mass,Nx,Ny,fs,Matrix,BC):
    '''
    Picard algorithm for NLTPFA and NLMPFA
    '''
    
    #Picard algo Initialization 
    
    f=np.ones((Nx*Ny))
    s=0#Picard algorithm iterator
    eps=1e-6 #stopping criteria
    norm_list=[]
    test_neg=False
                
    while True:
        
        f_1=f
        U1=f_1.reshape(len(y_cell),len(x_cell))
        U=mapping(U1,fs,x_all,y_all,x_cell,y_cell)        
        
        (diago_10_nl,diago_1_nl,diago0_nl,diago1_nl,diago10_nl)=Matrix(D1,D2,D12,x_all,y_all,mass,U)
        data_nl = [diago_10_nl,diago_1_nl,diago0_nl,diago1_nl,diago10_nl]
        offsets_nl = np.array([-(Nx),-1,0,1,(Nx)])
        E_nl=sp.dia_matrix((data_nl, offsets_nl), shape=((Nx)*(Ny),(Nx)*(Ny)))
        S1_nl=BC(D1,D2,D12,x_all,y_all,fs,U)
        
        RHS_nl=-S1_nl-S
        f=spsolve(E_nl,RHS_nl)
        
        s=s+1
        
        print('***NONLIN Iter : '+str(s))
        norm=(abs((f_1-f)).max()/abs(f_1).max())
        norm_list.append(norm)
        print('Norm = '+str(norm))
        print('Min = '+str(f.min()))
                
        if f.min()<0.0:
            test_neg=True
            print('PROBLEM f<0 nonlinear ')
            break
        
        if s>1 and norm_list[-2]<norm:
            print('RELAXATION')
            f=f*0.5+f_1*0.5
                
        if norm<eps or s>1000:
            nb_iter=s
            if s>1000:
                print('COULD NOT CONVERGE')
            break
    
    return(f,f.min(),f.max(),nb_iter,test_neg)


def Picard_algo_RNLMPFA(x_all,y_all,x_cell,y_cell,D1,D2,D12,mass,Nx,Ny,fs):
    '''
    Picard algorithm for NLTPFA and NLMPFA
    '''
    
    c1=min(c1c2(x_all,y_all,Dxx,Dyy,Dyx))/10.0
    c2=2.0*c1
        
    #Picard algo Initialization 
    
    f=np.ones((Nx*Ny))
    s=0#Picard algorithm iterator
    eps=1e-6 #stopping criteria
    norm_list=[]
    test_neg=False
    
    nord=[]
                
    while True:
        
        f_1=f
        U1=f_1.reshape(len(y_cell),len(x_cell))
        U=mapping(U1,fs,x_all,y_all,x_cell,y_cell)        
        
        (diago_1_1_nl,diago_10_nl,diago_11_nl,diago_1_nl,diago0_nl,diago1_nl,diago1_1_nl,diago10_nl,diago11_nl)=Matrix_RNLMPFA(D1,D2,D12,x_all,y_all,mass,U,c1,c2)
        data_nl = [diago_1_1_nl,diago_10_nl,diago_11_nl,diago_1_nl,diago0_nl,diago1_nl,diago1_1_nl,diago10_nl,diago11_nl]
        offsets_nl = np.array([-(Nx)-1,-(Nx),-(Nx)+1,-1,0,1,(Nx-1),(Nx),Nx+1])
        E_nl=sp.dia_matrix((data_nl, offsets_nl), shape=((Nx)*(Ny),(Nx)*(Ny)))
        S1_nl=BC_RNLMPFA(D1,D2,D12,x_all,y_all,fs,U,c1,c2)
        
        E_full=E_nl.todense()
        
        nord.append(verif_cond_nordbotten(x_all,y_all,E_full))
        
        RHS_nl=-S1_nl-S
        f=spsolve(E_nl,RHS_nl)
        
        s=s+1
        
        print('***NONLIN Iter : '+str(s))
        norm=(abs((f_1-f)).max()/abs(f_1).max())
        norm_list.append(norm)
        print('Norm = '+str(norm))
        print('Min = '+str(f.min()))
                
        if f.min()<0.0:
            test_neg=True
            print('PROBLEM f<0 nonlinear ')
            break
        
        if s>1 and norm_list[-2]<norm:
            print('RELAXATION')
            f=f*0.5+f_1*0.5
            
        if norm<eps or s>1000:
            nb_iter=s
            if s>1000:
                print('COULD NOT CONVERGE')
            break
    
    return(f,f.min(),f.max(),nb_iter,test_neg,nord)



### Computing
    
mesh_size=[20,40,80]

Err2_NLTPFA=[]

Err2_NLMPFA=[]

Err2_RNLMPFA=[]

nb_iter_NLTPFA=[]
norm_NLTPFA=[]

nb_iter_NLMPFA=[]
norm_NLMPFA=[]

nb_iter_RNLMPFA=[]
norm_RNLMPFA=[]

f_min_NLTPFA=[]
f_max_NLTPFA=[]

f_min_NLMPFA=[]
f_max_NLMPFA=[]

f_min_RNLMPFA=[]
f_max_RNLMPFA=[]
nord_list=[]

dx=[]
dy=[]

print('-----------------------------------------')

for N in mesh_size:
                            
    Nx=N #mesh size on dimension X
    Ny=N #mesh size on dimension Y
    
    print('Nx = '+str(Nx))
    print('Ny = '+str(Ny))
    print('Nu = '+str(Nx*Ny))

    #Omega domain frontiers
    lx=float(0.5)
    ly=float(0.5)
    ox=float(0.0)
    oy=float(0.0)
    
    #Mesh construction
    #Uniform cartesian grid
    x_all=np.linspace(ox,lx,2*Nx+1)
    y_all=np.linspace(oy,ly,2*Ny+1)
    #Cell centers
    x_cell=x_all[1:-1:2]
    y_cell=y_all[1:-1:2]
    
    dx.append(x_cell[1]-x_cell[0])
    
    #Cell sizes
    mass=np.array([(x_all[2*i]-x_all[2*i-2])*(y_all[2*j]-y_all[2*j-2]) for j in range(1,Ny+1,1) for i in range(1,Nx+1,1)])
        
    #Diffusion coefficients arrays   
    Dxx=np.array([[Dxx_2(a,b) for a in x_all] for b in y_all])
    Dyy=np.array([[Dyy_2(a,b) for a in x_all] for b in y_all])
    Dyx=np.array([[Dyx_2(a,b) for a in x_all] for b in y_all])
    
    if ox==0.0 and oy==0.0:
        Dxx[0,0]=0.5*(Dxx[1,0]+Dxx[0,1])
        Dyy[0,0]=0.5*(Dyy[1,0]+Dyy[0,1])
        Dyx[0,0]=0.5*(Dyx[1,0]+Dyx[0,1])
    
    #Source
    S=mass*np.array([sos(a,b) for b in y_cell for a in x_cell])
    
    #Analytical solution to evaluate the numerical error
    sol_ana=np.array([[fs(a,b) for a in x_cell]for b in y_cell])

    # #NLTPFA
    f,fmin,fmax,nb_iter,test_neg=Picard_algo_NLTPFA_NLMPFA(x_all,y_all,x_cell,y_cell,Dxx,Dyy,Dyx,mass,Nx,Ny,fs,Matrix_NLTPFA,BC_NLTPFA)                   
    f_min_NLTPFA.append(fmin)
    f_max_NLTPFA.append(fmax)
    nb_iter_NLTPFA.append(nb_iter)
    H=f.reshape((Ny,Nx))  
    Err2_NLTPFA.append(100*np.sqrt(simps(simps((H-sol_ana)**2,x_cell),y_cell)/simps(simps((sol_ana)**2,x_cell),y_cell)))   
    
    # #NLMPFA
    f,fmin,fmax,nb_iter,test_neg=Picard_algo_NLTPFA_NLMPFA(x_all,y_all,x_cell,y_cell,Dxx,Dyy,Dyx,mass,Nx,Ny,fs,Matrix_NLMPFA,BC_NLMPFA)                   
    f_min_NLMPFA.append(fmin)
    f_max_NLMPFA.append(fmax)
    nb_iter_NLMPFA.append(nb_iter)
    H=f.reshape((Ny,Nx))  
    Err2_NLMPFA.append(100*np.sqrt(simps(simps((H-sol_ana)**2,x_cell),y_cell)/simps(simps((sol_ana)**2,x_cell),y_cell)))
    
    # #RNLMPFA
    f,fmin,fmax,nb_iter,test_neg,nord=Picard_algo_RNLMPFA(x_all,y_all,x_cell,y_cell,Dxx,Dyy,Dyx,mass,Nx,Ny,fs)
    nord_list.append(max(nord))           
    f_min_RNLMPFA.append(fmin)
    f_max_RNLMPFA.append(fmax)
    nb_iter_RNLMPFA.append(nb_iter)
    H=f.reshape((Ny,Nx))  
    Err2_RNLMPFA.append(100*np.sqrt(simps(simps((H-sol_ana)**2,x_cell),y_cell)/simps(simps((sol_ana)**2,x_cell),y_cell)))
    
    print('-----------------------------------------')

print('iteration number for the NLTPFA scheme for each mesh size')
print(nb_iter_NLTPFA)
print('iteration number for the NLTPFA scheme for each mesh size')
print(nb_iter_NLMPFA)
print('iteration number for the NLTPFA scheme for each mesh size')
print(nb_iter_RNLMPFA)
print('number of cells that do not respect nordbotten conditions for each mesh size')
print(nord_list)
print('-----------------------------------------')

### Plots
plt.figure()
plt.subplot(211)
plt.plot(dx,Err2_NLTPFA,'o',label=r'$Err_{2}NLTPFA$')
plt.plot(dx,Err2_NLMPFA,'o',label=r'$Err_{2}NLMPFA$')
plt.plot(dx,Err2_RNLMPFA,'o',label=r'$Err_{2}R-NLMPFA$')
plt.plot(dx,Err2_NLTPFA[0]*dx[0]**-2*np.asarray(dx)**float(2),'--',label='$1/N^{2}$')
plt.plot(dx,Err2_NLTPFA[0]*dx[0]**-1*np.asarray(dx),'--',label=r'$1/N$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$dx$')
plt.ylabel('$Err$ %')
