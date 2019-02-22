import sys
import multiprocessing
import os
import hmf
import pickle
import math
import numpy as np
#import Functions_c
from numba import jit
import scipy.interpolate as inter
import matplotlib as mpl
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
import colossus.halo.mass_adv as massdefs
from colossus.lss import mass_function
from colossus.halo.mass_so import M_to_R
from astropy.cosmology import Planck15 as Cosmo_AstroPy
from halotools import empirical_models
from astropy import constants
from time import time
from halotools import empirical_models as EM
HM_SM = EM.Moster13SmHm(prim_haloprop_key = 'SM')
cosmology.setCosmology("millennium")
Cosmo = cosmology.getCurrent()
h = Cosmo.h



def StarFormationRate(SM, z,SFR_Model, ScatterOn =True):
    """
    Calculates Starformation rate
    Args:
    SM: Stellar Mass [log10 Msun]
    z: Redshift
    Returns:
    SFR: Star formation rate [log10 Msun yr-1]
    """
    
    #Schreiber 2015
    if SFR_Model == 'S15':
        m = SM-9; r = np.log10(1+z)    
        m0, a0, a1, m1, a2 = 0.5, 1.5, 0.3, 0.36, 2.5
        Max = m-m1-a2*r
        Max[Max<0] = 0

        log10MperY = m-m0+a0*r-a1*np.power(Max, 2)

    elif SFR_Model == 'T16':
       # Tomczak 2016 All Galaxies
        s0 = 0.195 + 1.157*(z) - 0.143*(z**2)
        logM0 = 9.244 + 0.753*(z) - 0.09*(z**2)
        Gamma = -1.118 #including -ve here to avoid it later
        log10MperY = s0 - np.log10(1 + np.power(np.power(10, (SM - logM0) ), Gamma))
        
    elif SFR_Model == "CE":
        
        s0 = 0.6 + 1.22*(z) - 0.2*(z**2)
        logM0 = 10.3 + 0.753*(z) - 0.15*(z**2)
        Gamma = -(1.3 - 0.1*(z))# - 0.03*(z[i]**2))#including -ve here to avoid it later
        log10MperY = s0 - np.log10(1 +np.power(np.power(10, (SM - logM0) ), Gamma))

    if ScatterOn:
        log10MperY = np.random.normal(log10MperY,0.2) # scatter in the MS
    else:
        return log10MperY
    
    return log10MperY
    #Tomczak 2016 All Galaxies
    #s0 = 0.195 + 1.157*(z) - 0.143*(z**2)
    #logM0 = 9.244 + 0.753*(z) - 0.09*(z**2)
    #Gamma = -1.118 #including -ve here to avoid it later
    #log10MperY = s0 - np.log10(1 + np.power(np.power(10, (SM - logM0) ), Gamma))
    #return log10MperY

def GetGasMass(SM, z, HM,SFR_Model):
    """
    Calculates Galaxy gas content
    Args:
        SM: Stellar Mass [log10 Msun]
        z: Redshift
        HM: Halo Mass [log10 Msun]
    Returns:
        Gas Mass [log10 Msun] 
    """


    
    #Used in Paper 1 #Stewart 2009
    #Calculates gass mass via SM scaling relation
    #alpha = -0.59*( (z + 1)**0.45 ) #minus here to avoid it later
    #GasMass = SM + np.log10(0.04) + alpha*(SM - 11.6532)
    
    #New relation using M*-SFR-Mgas proxy
    #GasMass = np.zeros(len(SM))
    #mlow = np.ma.masked_less(SM,10.5).mask
    #mhigh = np.ma.masked_greater_equal(SM,10.5).mask
    #GasMass[mhigh] = 9.22 + 0.81*StarFormationRate(SM[mhigh], z,SFR_Model, ScatterOn = False )
    #GasMass[mlow] = 7.9 +  1.7*StarFormationRate(SM[mlow], z,SFR_Model, ScatterOn = False)
     
    GasMass = 9.22 + 0.81*StarFormationRate(SM, z,SFR_Model, ScatterOn = False )
    GasMass = np.random.normal(GasMass,0.2)
    
    #Controls the maximum amount of mass in a DMHalo  
    MaxGas = np.full_like(GasMass, GetMaxGasMass(HM))
    GasMass[GasMass > MaxGas] = MaxGas[GasMass > MaxGas]

    return GasMass

def GetMaxGasMass(DM):
    """
    Calculates Maximum Gas Mass from cosmology relation
    Args:
        DM: Dark matter halo mass [log10 Msun] 
    Returns:
        Gas Mass [log10 Msun] 
    """
    return( DM + np.log10( ( Cosmo.Ob0 ) / ( Cosmo.Om0 ) ) )

def RedshiftToTimeArr(Redshift):
    """
    Dumb wrapper where I have replace outdated old code
    Args:
        Redshift: z 
    Returns:
        Time [Gyr]
    """
    Output = Cosmo.age(Redshift)
    return(Output)

def Get_HM_History(AnalyticHaloMass, AnalyticHaloMass_min, AnalyticHaloMass_max, AnalyticHaloBin):
    """
    Makes or loads halomass groth histories then sorts the arrays for the main program
    Args:
        AnalyticHaloMass: Array of halomasses (N,) [log10 Msun]
        AnalyticHaloMass_min: Minimum Halo Mass [log10 Msun]
        AnalyticHaloMass_max: Maximum Halo Mass [log10 Msun]
        AnalyticHaloBin: Halomass binwidth [dex]
    Returns:
        z: Redshifts for the program (M,)
        AvarageHaloMasses: Halomass histories for each bin in input AnalyticHaloMass (N, M) [log10 Msun]
    """
    #If we have created this array before load it else make it
    FileName = "{}{}{}{}.dat".format(AnalyticHaloMass_min, AnalyticHaloMass_max, AnalyticHaloBin, h)
    if FileName in os.listdir(path="./MasterArr"):
        AvaHaloMass_wz = np.loadtxt("./MasterArr/"+FileName)
    else:
        #We create the array like:
        #Redshift / M1 / M2 / M3 ...
        #   0      12  12.1  12.2 ...
        #runs N(20) concurent process of Halogrowth(M) creating and destroying files used/created by vandenbosch 14
        #output is the array AvaHaloMass_wz as detailed above
        with multiprocessing.Pool(processes = 20) as pool:
            PoolReturn = pool.map(Halogrowth, AnalyticHaloMass)
        Switch = True
        for z, M in PoolReturn:
            if Switch:
                AvaHaloMass_wz = np.column_stack((z, M))
                Switch = False
            else:
                AvaHaloMass_wz = np.column_stack((AvaHaloMass_wz, M))
            #print(AvaHaloMass_wz)

        #Sorts by the top row so we have z first then the mass arrays smallest to largest at z=0
        AvaHaloMass_wz = AvaHaloMass_wz[:,np.argsort(AvaHaloMass_wz[0])]
        #Units are Mvir h-1

        #Plots the massgrowth of out halos
        plt.figure(figsize = (9, 6), dpi = 200)
        for i in range(1, len(AvaHaloMass_wz[0])):
            ColorParam = len(AvaHaloMass_wz[0]) - 1
            if (i == 1) or (i == len(AvaHaloMass_wz[0]) -1) or i%10 == 0:
                plt.plot(np.log10(AvaHaloMass_wz[:,0] +1), AvaHaloMass_wz[:,i] - AvaHaloMass_wz[:,i][0], color = ( (ColorParam-i)/ColorParam, 0, (i-1)/ColorParam ), label = "{}".format(AvaHaloMass_wz[:,i][0]) )
            else:
                plt.plot(np.log10(AvaHaloMass_wz[:,0] +1), AvaHaloMass_wz[:,i] - AvaHaloMass_wz[:,i][0], color = ( (ColorParam-i)/ColorParam, 0, (i-1)/ColorParam ) )
        plt.xlabel("log[1+z]")
        plt.ylabel("log[M(z)/$M_0$]")
        plt.legend()
        plt.savefig("./Figures/HaloGrowth.png")
        plt.clf()
        #We now have a sorted halmass growth 2d array

        #Save this array to speed up subsequent runs
        np.savetxt("./MasterArr/"+FileName, AvaHaloMass_wz)
    z = AvaHaloMass_wz[:,0]
    AvaHaloMass = AvaHaloMass_wz[:,1:]
    
    #We here cut the arrays so the code stops at z=0.1 where we have SDSS data change this to go to 0.
    z_Cut_Bin = np.digitize(0.1, bins = z)
    return z[z_Cut_Bin:], AvaHaloMass_wz[z_Cut_Bin:]
    #Units are Mvir h-1

#Makes the HMF interpolation function using HMF calc
def Make_HMF_Interp_Old():
    """
    Makes or loads a halo mass function
    Returns:
        HMF_fun: a function that take Mhalo and redshift and returns a numberdensity
    """
    print("WARNING OLD HMF INSTEAD USE from colossus.lss import mass_function")
    if 'hmf_fun.pkl' in os.listdir():
        HMF_fun = pickle.load(open('hmf_fun_old.pkl', 'rb'))
    else:
        #Halo mass function from hmf
        #http://hmf.icrar.org/hmf_finder/form/create/
        #http://hmf.readthedocs.io/en/latest/index.html
        #Default cosmology is Planck15
        HMF_fit = hmf.fitting_functions.Tinker10

        #The mass and redshift range should be larger than the simulation
        #Mass
        Min_x =8; Max_x = 16; Step_x = 0.01
        HMF_x = np.arange(Min_x, Max_x, Step_x)
        #Redshift
        Min_y =0; Max_y = 7; Step_y = 0.01
        HMF_y = np.arange(Min_y, Max_y, Step_y)

        HMF = hmf.MassFunction(Mmin=Min_x-0.5, Mmax=Max_x+0.5, dlog10m=Step_x, hmf_model=HMF_fit, delta_h=200.0, delta_wrt='crit', delta_c=1.686)
        HMF.update(z = 0)
        #wants M/h in input, changes halo mass defnition from 200c to vir, WORKS ONLY WITH DIEMER&kRAVTSOV 2015
        Mvir = massdefs.changeMassDefinitionCModel(M=HMF.m, z=0, mdef_in='200c', mdef_out='vir')[0]

        Interp_Temp = inter.interp1d(np.log10(Mvir), HMF.dndlog10m)
        HMF_z = Interp_Temp(HMF_x)

        for z_step in HMF_y[1:]:
            HMF.update(z = z_step)
            #wants M/h in input, changes halo mass defnition from 200c to vir, WORKS ONLY WITH DIEMER&kRAVTSOV 2015
            Mvir = massdefs.changeMassDefinitionCModel(M=HMF.m, z=z_step, mdef_in='200c', mdef_out='vir')[0]
            Interp_Temp = inter.interp1d(np.log10(Mvir), HMF.dndlog10m)
            HMF_z = np.vstack((HMF_z, Interp_Temp(HMF_x)))

        HMF_fun = inter.interp2d(HMF_x, HMF_y, HMF_z)
        pickle.dump(HMF_fun, open('hmf_fun_old.pkl', 'wb'))
        
        
    return HMF_fun # differential mass function h^3 Mpc^-3 dex-1
#Makes the HMF interpolation function using HMF calc
def Make_HMF_Interp():
    """
    Makes or loads a halo mass function
    Returns:
        HMF_fun: a function that take Mhalo and redshift and returns a numberdensity
    """
    if 'hmf_fun.pkl' in os.listdir():
        HMF_fun = pickle.load(open('hmf_fun.pkl', 'rb'))
    else:
        #Halo mass function from hmf
        #http://hmf.icrar.org/hmf_finder/form/create/
        #http://hmf.readthedocs.io/en/latest/index.html
        #Default cosmology is Planck15
        HMF_fit = hmf.fitting_functions.Tinker10

        #The mass and redshift range should be larger than the simulation
        #Mass
        Min_x =8; Max_x = 16; Step_x = 0.01
        HMF_x = np.power(10,np.arange(Min_x, Max_x, Step_x))
        #Redshift
        Min_y =0; Max_y = 7; Step_y = 0.01
        HMF_y = np.arange(Min_y, Max_y, Step_y)
        
        HMF_z = mass_function.massFunction(HMF_x, HMF_y[0], mdef = 'vir', model = 'despali16', q_out='dndlnM')*np.log(10)
        for z_step in HMF_y[1:]:
            HMF_dndlog10m =  mass_function.massFunction(HMF_x, z_step, mdef = 'vir', model = 'despali16', q_out='dndlnM')*np.log(10)
            HMF_z = np.vstack((HMF_z, HMF_dndlog10m))

        HMF_fun = inter.interp2d(np.log10(HMF_x), HMF_y, HMF_z)
        pickle.dump(HMF_fun, open('hmf_fun.pkl', 'wb'))
        
        
    return HMF_fun # differential mass function h^3 Mpc^-3 dex-1

#Returns the Unevolved SHMF from Jiang, van den Bosch.
#Units are Mvir h-1
def dn_dlnX(Parameters, X):
    """
    Caculates subahlo mass funtions
    Args:
        Parameters: Dictonary containg 'gamma', 'alpha', 'beta', 'omega', 'a'
        X: m/M arrays desired subhalo/parenthalo
    Returns:
        dn_dlogX_arr: Numberdensitys per dex #N dex-1
    """
    Part1 = Parameters['gamma']*np.power(Parameters['a']*X, Parameters['alpha'])
    Part2 = np.exp(-Parameters['beta']*np.power(Parameters['a']*X, Parameters['omega']))
    dn_dlnX_arr = Part1*Part2
    dn_dlogX_arr = dn_dlnX_arr*2.30
    return dn_dlogX_arr #N dex-1

#
#Units are Mvir h-1
def Halogrowth(log_M_h, FullReturn = False):
    """
    Runs a multithread safe instance of the halo mass growth algorithm from Vandenbosch+ 2014
    Args:
        log_M_h: HaloMass at z = 0 (N,) [log10 Msun]
    Returns:
        z: redshift steps (M,)
        M_out: halomasses at redshift steps (N,M)[log10 Msun]
    """
    PID = log_M_h
    print(PID, end = "\r")
    #Input String is written to a file to be paees into VDB14 as paramaters
    Input_Str =("\
    0.307                                        ! Omega_0\n\
    0.678                                        ! h (= H_0/100)\n\
    0.823                                        ! sigma8\n\
    0.96                                         ! nspec\n\
    0.02298                                      ! Omega_b_h2\n\
    %.1E                                         ! M_0  (h^{-1} Msun)\n\
    0.0                                          ! z_0\n\
    1                                            ! median (0) or averages (1)\n\
    %s.dat                                       !Output File\n\
    " %(10**log_M_h, PID))
    with open('./VDB13/%s.in' %(PID), "w") as f:
        f.write(Input_Str)
    #starts the system command to run VDB14
    os.system("./VDB13/getPWGH < ./VDB13/%s.in" %(PID))
    #Loads the output of VDB14
    log_Mz_M0 = np.loadtxt("./%s.dat" %(PID))
    #Removes the file we made to run VDB14 and the file created by VDB14
    os.remove("./VDB13/%s.in"%(PID))
    os.remove("./%s.dat" %(PID))


    if FullReturn:
        return log_Mz_M0
    else:
        #Gets from the VDB14 output the redshift z and h corrected halomass array
        z = log_Mz_M0[:,1]
        M_out =(log_Mz_M0[:,3] + log_M_h)
        return (z, M_out)

#Starformation and other baryonic processes
def StarFormation(SM_Sat, TTZ0, Tdyf, z_infall, z_return, z_all, HM_infall, AvaHaloMass, Paramaters):
    """
    Creates arrays, varibles and sends correct parameters to the Cython accelerated Starformation routines
    Args:
        SM_Sat: Stellar mass at infall of satellite (1,) [log10 Msun]
        TTZ0: The time to redshift 0 (N,) [Gyr]
        Tdyf: Dynamical Friction timecale (1,) [Gyr]
        z_infall: Redshift of infall (1,)
        z_return: Redshift to evolve to
        z_all: the entire redshift array of the simulation
        MH_infall: Halomass at satellite infall [log10 Msun]
        AvaHaloMass: Halomass over the entire merge [log10 Msun]
        Paramaters: Dictonary containing 'SF_Q' an interger passed to slect starformation and quenching routines
    Returns:
        M_out: Stellarmass at each redshift (N,) [log10 Msun]
        sSFR: Specific starformation rate (N,) [yr -1]
    """   
    #Make sure we iterate from zmax to zmin
    z_all = np.flip(z_all, 0)
    AvaHaloMass = np.flip(AvaHaloMass, 0)
    #Create arays from the simulation start to the galaxy return
    z_bin_i = np.digitize(z_infall, z_all)
    z_bin_r = np.digitize(z_return, z_all)
    z_range = z_all[z_bin_i:z_bin_r]
    t = Cosmo.lookbackTime(z_all)
    d_t = t[1:] - t[:-1]
    d_t = np.insert(d_t, -1, d_t[-1])
    d_t = np.abs(d_t[z_bin_i:z_bin_r])
    t = t[z_bin_i:z_bin_r] 
    
    #Quenching, Wetzel+ 13
    Tau_f = -0.5*SM_Sat + 5.7
    Tau_f[Tau_f <= 0.2] = 0.2 # Fadetime
    Tau_d = 3.5 - (np.exp( (SM_Sat - 10.8)*2 ))
    Tau_d[Tau_d <= 1.0] = 1.0   
    #Host Dep, Fillingham+ 16       
    Host_Dep = (AvaHaloMass[0] -15)/5
    if Host_Dep < 0:
        Host_Dep = 0
    elif Host_Dep > 1:
        Host_Dep = 1
    Tau_d[SM_Sat < 9+Host_Dep] = 2.0
            
    T_quench = t[0] - Tau_d #Quenchtime

       
    # Maximum GasMass available is that at accretion
    MaxGas = np.power(10, GetGasMass(SM_Sat, z_all[z_bin_i], HM_infall,SFR_Model = str(Paramaters['SFR_Model'])))

    #Call the accelerated Cython Function
    M_out, M_dot, SFH, GMLR, GasMass = Functions_c.Starformation_c(SM_Sat, t, d_t, z_range, MaxGas, T_quench, Tau_f, z_infall = z_infall, SFR_Model = str(Paramaters['SFR_Model']))
    
    #Calculate the Specific Star formation Rate
    sSFR = np.log10(np.divide(np.array(SFH), d_t*np.power(10, 9)*np.power(10, np.array(M_out))))

    GasMass = np.array(GasMass)
    mask = np.ma.masked_equal(GasMass,-1).mask
    GasMass[mask] = np.log10(GasMass[mask])
    GasMass[np.logical_not(mask)] = np.nan
    
    return np.array(M_out), sSFR, GasMass

def Delta_vir(z):
    """
    Bryan and Norman 1998
    Args:
        Redshift: z
    Returns:
        Delta_vir
    """
    x = Cosmo.Om(z) - 1
    return (18*np.power(np.pi, 2) + 82*x - 39*np.power(x, 2)) #np.divide((18*np.power(np.pi, 2) + 82*x - 39*np.power(x, 2)), Cosmo.Om(z))


def HaloMassLoss_w(m, M, z, z_bin_r, z_bin_i):    
    """
    VandenBosch 2005 functioanl form Jiang 16 parameters
    Args:
        #m: is satilite mass on infall
        #M: is parent halo mass associated with each z
        #z: is list of redshift steps between infall and merge/return
    Returns:
        we return m and fstrip which are the satilite masses/stripping fractions at each z
    """
    t = RedshiftToTimeArr(z) #create times from the redshifts
    d_t = t[1:] - t[:-1] #create timesteps
    d_t = np.insert(d_t, -1, d_t[-1]) #insert timestep
    
    #Flip arrays to run in the corret time direction
    M, z, d_t = np.flip(M, 0), np.flip(z[z_bin_r:z_bin_i], 0), np.flip(d_t[z_bin_r:z_bin_i], 0)
    
    #Send to Cython accelerated function
    m_new = Functions_c.HaloMassLoss_c(m, M, z, d_t)
    
    #calaculate the fration stripped
    f_strip = (1 - np.power(10, m_new-m))
    f_strip[f_strip < 0.0] = 0.0
    
    return np.flip(m_new, 0), np.flip(f_strip, 0)

def StellarMassLoss(HM_c, HM_s, SM, TTZ0, Tdyn):
    """
    Args:
        HM_c: Halomass Central [log10 Msun]
        HM_s: Halomass Satellite [log10 Msun]
        SM: Stellarmass [log10 Msun]
        TTZ0: Time to redshift zero [Gyr]
        Tdyn: Timescale of infall [Gyr]
    Returns:
        Stripped SM [log10 Msun]
    """
    #Check that galaxy merges before redshift 0 else create a factor reducing the stripping
    if TTZ0 < Tdyn:
        Factor = TTZ0/Tdyn
    else:
        Factor = 1
    #halo mass ratio converted to natural numbers
    Mh_Ms = np.power(10, HM_c - HM_s)
    #Catteneo++ 2011
    Strip = np.power(0.6, (1.428/(2*np.pi))*(Mh_Ms/np.log(1+Mh_Ms)))
    #Correct for factor
    Strip_f = Strip + (1-Strip)*(1-Factor)
    return np.log10(Strip_f)+SM #Return stripped SM

def DynamicalTime_Fun(Redshift):
    """
    Args:
        Redshift: z
    Returns:
        Tdyn in Gyr
    """
    G_MPC_Msun_Gyr = -14.34693206746732 #np.log10(constants.G.value*np.power(constants.pc.value*10**6,-3)*(constants.M_sun.value)*np.power(3.15576*10**16, 2))
    DynamicalTime = 1.628*(h**-1)*np.power(Delta_vir(Redshift)/178, -0.5)*np.power(Cosmo.Hz(Redshift)/Cosmo.H0, -1) #Jiang & VDB 2016 #Gyr

    return DynamicalTime

def DynamicalFriction(HostHaloMass, SatiliteHaloMass, Redshift, Paramaters):
    """
    Args:
        HostHaloMass: [log10 Mvir h-1]
        SatiliteHaloMass: [log10 Mvir h-1]
        Redshift: z
        Paramaters: Dictonary containing 'AltDynamicalTime' a float.
    Returns:
        Infall timescale in Gyr
    """
    
    #Constants to paramatize dependance on Mhost/Msat from McCavana 2012   
    #double A = 0.216, B = 1.3, C = 1.9, D = 1.0 #B-K values
    A = 0.9; B = 1.0; C = 0.6; D = 0.1 #McCavana values
    MassRatio = 10**(HostHaloMass - SatiliteHaloMass) #unitless
    VR = np.divide(M_to_R(10**HostHaloMass, Redshift, 'vir'), h*(10**3)) #mega parsecs
    Tdyn = DynamicalTime_Fun(Redshift) #Gyr
    if Paramaters['AltDynamicalTime'] != False:
        Tdyn = Tdyn * Paramaters['AltDynamicalTime']
    
    NormalRnd = 0.5 #We are considering only avarage halos on circular orbits

    """
    NormalRnd = np.random.normal(loc = 0.5, scale = 0.23) #Khochfar & Burkert 2006
    if( NormalRnd >=1 ): ##these just keep NormalRnd in check
        NormalRnd = 0.999
    elif( NormalRnd <= 0 ):
        NormalRnd = 0.001
    """

    OrbitalEnergy = (VR * NormalRnd**2.17) / (1 - np.sqrt( 1 - (NormalRnd**2) ) ) #energy
    #using formula in Boylan & kolchin 2008
    Part1 = MassRatio**B
    Part2 = np.log( 1 + MassRatio )
    Part3 = np.exp( C * NormalRnd ) #exponent of orbital circularity
    Part4 = ( OrbitalEnergy/VR )**D #(OrbitalEnergy/virialradius)^D
    return (Tdyn*A*(Part1/Part2)*Part3*Part4) #returns a timescale of infall



def SizesK13(DM,z_in,ScatterSizeOn=False,Scatter=0.01):
    
    ''' This functions assigns sizes for satellites at infall (Hearin+17) according to their dark matter halo size (Kravtsov 2013)
        It is assumed that these do not change during the time before merging
        This implies that if stripping is on this recipe must be reviewed
    '''
    np.random.seed(int(time()+os.getpid()*1000))
    
    HaloSize = M_to_R(10**(DM),mdef='vir',z=z_in)/h
    Normalization = 0.018 # might actually change with z (Somerville+18)
    StellarSize =  Normalization*HaloSize
    
    #Adding Scatter
    if(ScatterSizeOn):
        
        logSize = np.random.normal( np.log10(StellarSize),scale = Scatter, size = np.shape(SM))
        
    return logSize






##DarkMatterToStellarMassStart #moster 2013
##DarkMatterToStellarMassStart #moster 2013
@jit
def DarkMatterToStellarMass(DM, z, Paramaters, ScatterOn = False, Scatter = 0.001, Pairwise = True):
    """ 
    This funtion returns Stellar mass in log10 Msun, all arguments should be passed in simmilar cosmology (Planck 15 unless otherwise stated)
    DM and z is longer than 1 are assumed pairwise if N == M, otherwise Array N is calculated for all elements (M) of z. If N==M but pairwise is not desired pass Pairwise == False
    Args:
        DM: Dark Matter in log10 Msun. Can be (1,), (N,), or (N, M)).
        z: Redshift. Can be (1,) or (M,) 
        Parameters: Python Dictonary Containing Subdictonary 'AbnMtch':
                        Containing Booleans: 'z_Evo', 'Moster', 'Override_0', 'Override_z', 'G18'
                        Containing Parameters: 'Scatter'
                        Containing Dictonary: 'OverRide':
                            Containing Parameters: 'M10', 'SHMnorm10', 'beta10', 'gamma10', 'M11', 'SHMnorm11', 'beta11', 'gamma11' 
        ScatterOn: Bool to switch scatter on/off
        Scatter: Scatter set low should be set in parameter section or sent via Scatter in dictonary.
        Pairwise: Bool. If true and N==M n and M will not be calculated Pairwise
    Returns:
        Stellar mass array in log10 Msun. Shape will be (1,), (N,), or (N, M) depending on the sape of inputs.
    Raises: 
        N/A
    """
    np.random.seed(int(time()+os.getpid()*1000))
    Paramaters = Paramaters['AbnMtch']
    if Paramaters['z_Evo']:
        if Paramaters['Moster']:
            zparameter = np.divide(z, z+1)
        elif Paramaters['Override_0'] or Paramaters['Override_z'] or Paramaters['G18'] or Paramaters['G18_notSE']:
            zparameter = np.divide(z-0.1, z+1)
        else:
            zparameter = np.divide(z-0.1, z+1)
    else:
        zparameter = 0

    if ScatterOn == True:
        Scatter = Paramaters['Scatter']
    
    if Paramaters['Override_0'] or Paramaters['Override_z']:
        Override = Paramaters['Override']
    
    #parameters from moster 2013
    if(Paramaters['Moster']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.590, 0.0351, 1.376, 0.608, 0.15
        M11, SHMnorm11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329
    #paremeters from centrals Paper1
    if(Paramaters['G18']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.95, 0.032, 1.61, 0.54, 0.11
        M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
    #paremeters from centrals Paper1 with a slight (-0.15) correction away from sersicexp
    if(Paramaters['G18_notSE']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.95, 0.032, 1.61, 0.62, 0.11 #12.00, 0.022, 1.56, 0.55, 0.15
        M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, 0.0 #0.4, 0.0, -0.5, 0.1
    #allows user to sent in their own abundance matching parameters either fixed at redshift 0/0.1 or evolving 
    if(Paramaters['Override_0']):
        M10, SHMnorm10, beta10, gamma10 = Override['M10'], Override['SHMnorm10'], Override['beta10'], Override['gamma10']
        M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1 #1.195, -0.0247, -0.826, 0.329
    if(Paramaters['Override_z']):
        M10, SHMnorm10, beta10, gamma10 = Override['M10'], Override['SHMnorm10'], Override['beta10'], Override['gamma10']
        M11, SHMnorm11, beta11, gamma11 = Override['M11'], Override['SHMnorm11'], Override['beta11'], Override['gamma11']
    #For Pairfraction Testing
    if Paramaters['PFT']:
        if(Paramaters['M_PFT1']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.7, 0.032, 1.61, 0.54, 0.11 #M10-0.25
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
        if(Paramaters['M_PFT2']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.5, -0.02, -0.6, -0.1 #M11 + 0.1
        if(Paramaters['M_PFT3']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.3, -0.02, -0.6, -0.1 #M11 - 0.1
        if(Paramaters['N_PFT1']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.036, 1.61, 0.54, 0.11 #N10 +0.04
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
        if(Paramaters['N_PFT2']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.013, -0.6, -0.1 #N11 + 0.07
        if(Paramaters['N_PFT3']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.027, -0.6, -0.1 #N11 - 0.07
        if(Paramaters['b_PFT1']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.36, 0.54, 0.11 #beta10 - 0.3
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
        if(Paramaters['b_PFT2']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.3, -0.1 #beta11 + 0.3
        if(Paramaters['b_PFT3']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.9, -0.1 #beta11 - 0.3
        if(Paramaters['g_PFT1']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.60, 0.11 #gamma10 + 0.06
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
        if(Paramaters['g_PFT2']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, 0.1 #gamma11 + 0.2
        if(Paramaters['g_PFT3']):
            M10, SHMnorm10, beta10, gamma10, Scatter= 11.95, 0.032, 1.61, 0.54, 0.11
            M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.3 #gamma11 - 0.2
    
    #putting the parameters together for inclusion in the Moster 2010 equation
    M = M10 + M11*zparameter
    N = SHMnorm10 + SHMnorm11*zparameter
    b = beta10 + beta11*zparameter
    g = gamma10 + gamma11*zparameter

    # Moster 2010 eq2
    if ((np.shape(DM) == np.shape(z)) or np.shape(z) == (1,) or np.shape(z) == ()) and Pairwise:
        SM =  np.power(10, DM) * (2*N*np.power( (np.power(np.power(10,DM-M), -b) + np.power(np.power(10,DM-M), g)), -1))
    else:
        if (np.shape(DM)[0] != np.shape(z)[0]):
            M = np.full((np.size(DM), np.size(z)), M).T
            N = np.full((np.size(DM), np.size(z)), N).T
            b = np.full((np.size(DM), np.size(z)), b).T
            g = np.full((np.size(DM), np.size(z)), g).T
            DM = np.full((np.size(z), np.size(DM)), DM)
        SM =  np.power(10, DM) * (2*N*np.power( (np.power(np.power(10,DM-M), -b) + np.power(np.power(10,DM-M), g)), -1))
    #Adding Scatter
    if(ScatterOn):
        Scatter_Arr = np.random.normal(scale = Scatter, size = np.shape(SM))
        return( np.log10(SM) + Scatter_Arr)
    else:
        return( np.log10(SM))

def DarkMatterToStellarMass_Alt(DarkMatter, Redshift, Paramaters, ScatterOn = False, Scatter = 0.001):
    np.random.seed()
    Paramaters = Paramaters['AbnMtch']
    z = Redshift
    if(Paramaters['Behroozi18']):
        if(Paramaters['B18c']):
            """e = np.array([-1.480, 0.693, 0.042, 0.121])
            M = np.array([12.004, 2.220,2.229,-0.363])
            alpha = np.array([2.041,-1.276,-1.082,0.147])
            beta = np.array([0.500,-0.219,-0.168,0.305])
            gamma = np.array([-0.848,-2.115,-0.664])
            delta = 0.350"""
            e = np.array([-1.340,0.404,-0.048,0.133])
            M = np.array([12.027,2.582,2.594,-0.409])
            alpha = np.array([1.999,-1.710,-1.393,0.192])
            beta = np.array([0.502,-0.267,-0.197])
            gamma = np.array([-0.788,-1.947,-0.658])
            delta = 0.340
        elif(Paramaters['B18t']):
            """e = np.array([-1.505,0.607,0.002,0.124])         
            M = np.array([11.979,2.293,2.393,-0.380])
            alpha = np.array([1.998,-1.394,-1.175,0.166])
            beta = np.array([0.512,-0.181,-0.160])
            gamma = np.array([-0.738,-1.697,-0.573])
            delta = 0.382"""
            e = np.array([-1.357,0.139,-0.230,0.157])
            M = np.array([11.968,2.231,2.359,-0.374])
            alpha = np.array([2.025,-1.365,-1.174,0.167])
            beta = np.array([0.520,-0.135,-0.161])
            gamma = np.array([-0.729,-1.764,-0.639])
            delta = 0.351

        a      = 1/(1+Redshift)
        afac   = a-1


        log10_M  = M[0]     + (M[1]*afac)     - (M[2]*np.log(a))     + (M[3]*z)
        e_       = e[0]     + (e[1]*afac)     - (e[2]*np.log(a))     + (e[3]*z)
        alpha_   = alpha[0] + (alpha[1]*afac) - (alpha[2]*np.log(a)) + (alpha[3]*z)
        beta_    = beta[0]  + (beta[1]*afac)                         + (beta[2]*z)
        log10_g  = gamma[0] + (gamma[1]*afac)                        + (gamma[2]*z)


        x = DarkMatter-log10_M
        gamma_ = np.power(10, log10_g)

        Part1 = np.log10(np.power(10, -alpha_*x) + np.power(10, -beta_*x))
        Part2 = np.exp(-0.5*np.power(np.divide(x, delta), 2))
        M_Star = log10_M+(e_ - Part1 + gamma_*Part2)


        if ScatterOn:
            Scatter = np.random.normal(scale = Scatter, size = np.shape(M_Star))
            return M_Star + Scatter
        else:
            return M_Star
                        
    elif Paramaters['Behroozi13'] or Paramaters['Lorenzo18']:
        if(Paramaters['Behroozi13']):
            e = np.array([-1.777, -0.006, 0.000, -0.119])
            M = np.array([11.514,-1.793,-0.251])
            alpha = np.array([-1.412,0.731])
            delta = np.array([3.508,2.608,-0.043])
            gamma_b = np.array([0.361,1.391,0.279])
            ep = np.array([0.218,-0.023])
        if(Paramaters['Lorenzo18']):
            e = np.array([-1.6695, -0.006, 0.000, -0.119])
            M = np.array([11.6097,-1.793,-0.251])
            alpha = np.array([-1.998,0.731])
            delta = np.array([3.2108,2.608,-0.043])
            gamma_b = np.array([0.4222,1.391,0.279])
            ep = np.array([0.1346,-0.023])
        a      = 1/(1+Redshift)
        afac   = a-1
        v_     = np.exp(-4*np.power(a,2))
        M_     = M[0]       + (M[1]      *afac +M[2]*z      )*v_
        e_     = e[0]       + (e[1]      *afac +e[2]*z      )*v_ + e[3]*afac
        alpha_ = alpha[0]   + (alpha[1]  *afac              )*v_
        delta_ = delta[0]   + (delta[1]  *afac +delta[2]*z  )*v_
        gamma_ = gamma_b[0] + (gamma_b[1]*afac +gamma_b[2]*z)*v_
        ep_    = ep[0]      + (ep[1]*afac                   )


        e_ = np.power(10, e_)
        M_ = np.power(10, M_)

        def f(x, a = alpha_, d = delta_, g = gamma_):
            Part1 = np.log10(np.power(10, a*x) + 1)
            Part2 = np.power(np.log10(1+np.exp(x)),g)
            Part3 = 1 + np.exp(np.power(10, -x))
            return -Part1 + d*np.divide(Part2, Part3)

        Part1 = np.log10(e_*M_)
        Part2 = f( np.log10( np.divide(np.power(10,DarkMatter), M_) ) )
        Part3 = f(0)

        M_Star = Part1 + Part2 - Part3
        Scatter = np.random.normal(scale = Scatter, size = np.shape(M_Star))
        if ScatterOn:
            return M_Star + Scatter
        else:
            return M_Star
    
    
def DM_to_SM(SMF_X, HMF, Halo_MR, HMF_Bin, SMF_Bin, Paramaters, Redshift = 0, N = 5000, UseAlt = False):
    """   
    Args:
        SMF_X: Stellar Mass Function Mass Range log10[$M_\odot$]
        HMF: Halo Mass Function Weights log10[ $\Phi$ Mpc^{-3} h^3]
        Halo_MR: Halo Mass Function Mass Range log10[$M_\odot$ h^{-1}]
        HMF_Bin: Binwidth of Halo_MR
        SMF_Bin: Binwidth of SMF_X
        Parameters: Dictonary of thing to pass to DarkMatterToStellarMass see afformentioned for details
        Redshift: z
        N: number of times to use
        UseAlt: Bool To switch to other Alt DM_to_SM
    Returns:
        SMF_X: Stellar Mass Function Mass Range log10[$M_\odot$], SMF numberdensties Phi [Mpc^-3] 
    """

    DM_In = np.repeat(Halo_MR - np.log10(h), N) #log Mh [Msun]
    Wt = np.repeat(np.divide(np.power(10, HMF + 3*np.log10(h))*HMF_Bin, N), N) #Phi/N [Mpc^-3]
    if UseAlt:
        SM = DarkMatterToStellarMass_Alt(DM_In, Redshift, Paramaters, ScatterOn = True) #log M* [Msun]
    else:
        SM = DarkMatterToStellarMass(DM_In, Redshift, Paramaters, ScatterOn = True) #log M* [Msun]
    
    SMF_Y, Bin_Edge = np.histogram(SM, bins = np.append(SMF_X, SMF_X[-1]+SMF_Bin), weights = Wt) #Phi [Mpc^-3], M* [Msun]
    
    return SMF_X, np.log10(np.divide(SMF_Y, SMF_Bin)) #M* [Msun], Phi [Mpc^-3] 


#For adding Scatter to match Data
def Gauss_Scatt(X, Y, Scatt = 0.1):
    """
    Args:
        X: data on x
        Y: data on Y
        Scatter: standard deviation to apply
    Returns:
        Scatterd X, and Y
    """
    N = 10000
    X_Bin = X[-1]-X[-2]
    DM_In = np.repeat(X, N)
    Wt = np.repeat(np.divide(Y, N), N)
    DM_Out = np.random.normal(DM_In, Scatt)
    Y_Out, X_Out = np.histogram(DM_Out, bins = np.append(X, X[-1] + X_Bin)-(X_Bin/2), weights = Wt)
    return X_Out[:-1]+(X_Bin/2), Y_Out


#==========================Saving Output========================================
def PrepareToSave(RunParam_List):
    os.system("rm DataOutput/StrippingOtp.dat")
    os.system("touch DataOutput/StrippingOtp.dat")
    os.system("echo 'M_zinf, Mzret, m, m_new, f_strip, z_inf, z_ret' >> ./DataOutput/StrippingOtp.dat")
    
    for RunParam in RunParam_List:
        os.system("rm -r DataOutput/RunParam_{}".format("".join(("{}_".format(i) for i in RunParam))))
    for RunParam in RunParam_List:
        os.system("mkdir DataOutput/RunParam_{}".format("".join(("{}_".format(i) for i in RunParam))))
def SaveData_3(AvaHaloMass, AnalyticalModel_SMF, Surviving_Sat_SMF_MassRange, RunParam):
    """Figure 3"""
    np.save("./DataOutput/RunParam_{}/Figure3_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Figure3_AnalyticalModel_SMF.npy".format("".join(("{}_".format(i) for i in RunParam))), AnalyticalModel_SMF)
    np.save("./DataOutput/RunParam_{}/Figure3_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
def SaveData_4_6(AvaHaloMass, AnalyticalModelFrac_, AnalyticalModelNoFrac_, SM_Cuts, RunParam):
    """Figure 4 + 6"""
    np.save("./DataOutput/RunParam_{}/Figure4_6_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Figure4_6_AnalyticalModelFrac_.npy".format("".join(("{}_".format(i) for i in RunParam))), AnalyticalModelFrac_)
    np.save("./DataOutput/RunParam_{}/Figure4_6_AnalyticalModelNoFrac_.npy".format("".join(("{}_".format(i) for i in RunParam))), AnalyticalModelNoFrac_)
    np.save("./DataOutput/RunParam_{}/Figure4_6_SM_Cuts.npy".format("".join(("{}_".format(i) for i in RunParam))), SM_Cuts)
def SaveData_5(Sat_SMHM, Sat_Parent_SMHM, RunParam):
    """Figure 5"""
    np.save("./DataOutput/RunParam_{}/Figure5_Sat.npy".format("".join(("{}_".format(i) for i in RunParam))), Sat_SMHM)
    np.save("./DataOutput/RunParam_{}/Figure5_Sat_Parent_SMHM.npy".format("".join(("{}_".format(i) for i in RunParam))), Sat_Parent_SMHM)
def SaveData_7(Mergers, Minor_Mergers, z, RunParam):
    """Figure 7"""
    np.save("./DataOutput/RunParam_{}/Figure7_Mergers.npy".format("".join(("{}_".format(i) for i in RunParam))), Mergers)
    np.save("./DataOutput/RunParam_{}/Figure7_Minor_Mergers.npy".format("".join(("{}_".format(i) for i in RunParam))), Minor_Mergers)
    np.save("./DataOutput/RunParam_{}/Figure7_z.npy".format("".join(("{}_".format(i) for i in RunParam))), Minor_Mergers)
def SaveData_8(AvaHaloMass, P_Elliptical, RunParam):
    """Figure 8"""
    np.save("./DataOutput/RunParam_{}/Figure8_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Figure8_P_Elliptical.npy".format("".join(("{}_".format(i) for i in RunParam))), P_Elliptical)
def SaveData_9(AvaHaloMass, z, Analyticalmodel_SI, RunParam):
    """Figure 9"""
    np.save("./DataOutput/RunParam_{}/Figure9_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Figure9_z.npy".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/Figure9_Analyticalmodel_SI.npy".format("".join(("{}_".format(i) for i in RunParam))), Analyticalmodel_SI)
def SaveData_10(AvaHaloMass, AnalyticalModel_SMF, Surviving_Sat_SMF_MassRange, RunParam):
    """Figure 10"""
    np.save("./DataOutput/RunParam_{}/Figure10_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Figure10_AnalyticalModel_SMF.npy".format("".join(("{}_".format(i) for i in RunParam))), AnalyticalModel_SMF)
    np.save("./DataOutput/RunParam_{}/Figure10_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
def SaveData_SMFhz(AvaHaloMass, AnalyticalModel_SMF_Highz, Surviving_Sat_SMF_MassRange, RunParam):
    """Figure 3"""
    np.save("./DataOutput/RunParam_{}/SMFhz_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/SMFhz_AnalyticalModel_SMF_Highz.npy".format("".join(("{}_".format(i) for i in RunParam))), AnalyticalModel_SMF_Highz)
    np.save("./DataOutput/RunParam_{}/SMFhz_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
def SaveData_z_infall(Surviving_Sat_SMF_MassRange, z, z_infall, RunParam):
    np.save("./DataOutput/RunParam_{}/z_infall_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
    np.save("./DataOutput/RunParam_{}/z_infall_z.npy".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/z_infall.npy".format("".join(("{}_".format(i) for i in RunParam))), z_infall)
def SaveData_sSFR(Surviving_Sat_SMF_MassRange, sSFR_Range, Satilite_sSFR, RunParam):
    np.save("./DataOutput/RunParam_{}/sSFR_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
    np.save("./DataOutput/RunParam_{}/sSFR_Range.npy".format("".join(("{}_".format(i) for i in RunParam))), sSFR_Range)
    np.save("./DataOutput/RunParam_{}/Satellite_sSFR.npy".format("".join(("{}_".format(i) for i in RunParam))), Satilite_sSFR)
def SaveData_Sat_SMHM(z, SatHaloMass, AvaHaloMass, Surviving_Sat_SMF_MassRange, Sat_SMHM, Sat_SMHM_Host, RunParam):
    np.save("./DataOutput/RunParam_{}/Sat_SMHM_z.npy".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/Sat_SMHM_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), SatHaloMass)
    np.save("./DataOutput/RunParam_{}/Sat_SMHM_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Sat_SMHM_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
    np.save("./DataOutput/RunParam_{}/Sat_SMHM_Sat_SMHM.npy".format("".join(("{}_".format(i) for i in RunParam))), Sat_SMHM)
    np.save("./DataOutput/RunParam_{}/Sat_SMHM_Sat_SMHM_Host.npy".format("".join(("{}_".format(i) for i in RunParam))), Sat_SMHM_Host)
def SaveData_Mergers(Accretion_History, z, AvaHaloMass, Surviving_Sat_SMF_MassRange, RunParam):
    np.save("./DataOutput/RunParam_{}/Mergers_Accretion_History".format("".join(("{}_".format(i) for i in RunParam))), Accretion_History)
    np.save("./DataOutput/RunParam_{}/Mergers_z".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/Mergers_AvaHaloMass".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Mergers_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
def SaveData_Pair_Frac(Pair_Frac, z, AvaHaloMass, Surviving_Sat_SMF_MassRange, RunParam):
    np.save("./DataOutput/RunParam_{}/Pair_Frac_Pair_Frac".format("".join(("{}_".format(i) for i in RunParam))), Pair_Frac)
    np.save("./DataOutput/RunParam_{}/Pair_Frac_z".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/Pair_Frac_AvaHaloMass".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Pair_Frac_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
def SaveData_Sat_Env_Highz(AvaHaloMass, z, AnalyticalModel_Cuts_Frac_highz, AnalyticalModel_Cuts_NoFrac_highz, SM_Cuts, RunParam):
    np.save("./DataOutput/RunParam_{}/Sat_Env_Highz_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Sat_Env_Highz_z".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/Sat_Env_Highz_AnalyticalModelFracHighz.npy".format("".join(("{}_".format(i) for i in RunParam))), AnalyticalModel_Cuts_Frac_highz)
    np.save("./DataOutput/RunParam_{}/Sat_Env_Highz_AnalyticalModelNoFracHighz.npy".format("".join(("{}_".format(i) for i in RunParam))), AnalyticalModel_Cuts_NoFrac_highz)
    np.save("./DataOutput/RunParam_{}/Sat_Env_Highz_SM_Cuts.npy".format("".join(("{}_".format(i) for i in RunParam))), SM_Cuts)
def SaveData_Raw_Richness(AvaHaloMass, z, Surviving_Sat_SMF_MassRange, Surviving_Sat_SMF_Weighting_highz, RunParam):
    np.save("./DataOutput/RunParam_{}/Raw_Richness_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Raw_Richness_Highz_z".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/Raw_Richness_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_MassRange)
    np.save("./DataOutput/RunParam_{}/Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy".format("".join(("{}_".format(i) for i in RunParam))), Surviving_Sat_SMF_Weighting_highz)
def SaveData_MultiEpoch_SubHalos(z, SatHaloMass, SurvivingSubhalos_z_z, RunParam):
    np.save("./DataOutput/RunParam_{}/MultiEpoch_SubHalos_z.npy".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/MultiEpoch_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), SatHaloMass)
    np.save("./DataOutput/RunParam_{}/MultiEpoch_SurvivingSubhalos_z_z.npy".format("".join(("{}_".format(i) for i in RunParam))), SurvivingSubhalos_z_z)
def SaveData_Pair_Frac_Halo(Pair_Frac_Halo, Accretion_History_Halo, z, AvaHaloMass, SatHaloMass, RunParam):
    np.save("./DataOutput/RunParam_{}/Pair_Frac_Halo_z.npy".format("".join(("{}_".format(i) for i in RunParam))), z)
    np.save("./DataOutput/RunParam_{}/Pair_Frac_Halo_Pair_Frac_Halo.npy".format("".join(("{}_".format(i) for i in RunParam))), Pair_Frac_Halo)
    np.save("./DataOutput/RunParam_{}/Pair_Frac_Halo_Accretion_History_Halo.npy".format("".join(("{}_".format(i) for i in RunParam))), Accretion_History_Halo)
    np.save("./DataOutput/RunParam_{}/Pair_Frac_Halo_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), AvaHaloMass)
    np.save("./DataOutput/RunParam_{}/Pair_Frac_Halo_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))), SatHaloMass)
#==========================Loading Output=======================================
def LoadData_3(RunParam_List):
    """Figure 3"""
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Figure3_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/Figure3_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    AnalyticalModel_SMF = []
    for RunParam in RunParam_List:
        AnalyticalModel_SMF.append(np.load("./DataOutput/RunParam_{}/Figure3_AnalyticalModel_SMF.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return AvaHaloMass, np.array(AnalyticalModel_SMF), Surviving_Sat_SMF_MassRange
def LoadData_4_6(RunParam_List):
    """Figure 4 + 6"""
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Figure4_6_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    SM_Cuts = np.load("./DataOutput/RunParam_{}/Figure4_6_SM_Cuts.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    AnalyticalModelFrac_ = []
    for RunParam in RunParam_List:
        AnalyticalModelFrac_.append(np.load("./DataOutput/RunParam_{}/Figure4_6_AnalyticalModelFrac_.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    AnalyticalModelNoFrac_ = []
    for RunParam in RunParam_List:
        AnalyticalModelNoFrac_.append(np.load("./DataOutput/RunParam_{}/Figure4_6_AnalyticalModelNoFrac_.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return AvaHaloMass, np.array(AnalyticalModelFrac_), np.array(AnalyticalModelNoFrac_), SM_Cuts
def LoadData_5(RunParam_List):
    """Figure 5"""
    Sat_SMHM = []
    Sat_Parent_SMHM = []
    for RunParam in RunParam_List:
        Sat_SMHM.append(np.load("./DataOutput/RunParam_{}/Figure5_Sat.npy".format("".join(("{}_".format(i) for i in RunParam)))))
        Sat_Parent_SMHM.append(np.load("./DataOutput/RunParam_{}/Figure5_Sat_Parent_SMHM.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return Sat_SMHM, Sat_Parent_SMHM
def LoadData_7(RunParam):
    """Figure 7"""
    Mergers = np.load("./DataOutput/RunParam_{}/Figure7_Mergers.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Minor_Mergers = np.load("./DataOutput/RunParam_{}/Figure7_Minor_Mergers.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load("./DataOutput/RunParam_{}/Figure7_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Mergers, Minor_Mergers, z
def LoadData_8(RunParam):
    """Figure 8"""
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Figure8_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    P_Elliptical = np.load("./DataOutput/RunParam_{}/Figure8_P_Elliptical.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, P_Elliptical
def LoadData_9(RunParam):
    """Figure 9"""
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Figure9_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load("./DataOutput/RunParam_{}/Figure9_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Analyticalmodel_SI = np.load("./DataOutput/RunParam_{}/Figure9_Analyticalmodel_SI.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, z, Analyticalmodel_SI
def LoadData_10(RunParam_List):
    """Figure 10"""
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Figure10_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/Figure10_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    AnalyticalModel_SMF = []
    for RunParam in RunParam_List:
        AnalyticalModel_SMF.append(np.load("./DataOutput/RunParam_{}/Figure10_AnalyticalModel_SMF.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return AvaHaloMass, np.array(AnalyticalModel_SMF), Surviving_Sat_SMF_MassRange
def LoadData_SMFhz(RunParam_List):
    """Figure 3"""
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/SMFhz_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/SMFhz_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    AnalyticalModel_SMF = []
    for RunParam in RunParam_List:
        AnalyticalModel_SMF.append(np.load("./DataOutput/RunParam_{}/SMFhz_AnalyticalModel_SMF_Highz.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return AvaHaloMass, np.array(AnalyticalModel_SMF), Surviving_Sat_SMF_MassRange
def LoadData_z_infall(RunParam):
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/z_infall_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load("./DataOutput/RunParam_{}/z_infall_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z_infall = np.load("./DataOutput/RunParam_{}/z_infall.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Surviving_Sat_SMF_MassRange, z, z_infall
def LoadData_sSFR(RunParam):
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/sSFR_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    sSFR_Range = np.load("./DataOutput/RunParam_{}/sSFR_Range.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Satellite_sSFR = np.load("./DataOutput/RunParam_{}/Satellite_sSFR.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Surviving_Sat_SMF_MassRange, sSFR_Range, Satellite_sSFR
def LoadData_Sat_SMHM(RunParam):
    z = np.load("./DataOutput/RunParam_{}/Sat_SMHM_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SatHaloMass = np.load("./DataOutput/RunParam_{}/Sat_SMHM_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Sat_SMHM_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/Sat_SMHM_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Sat_SMHM = np.load("./DataOutput/RunParam_{}/Sat_SMHM_Sat_SMHM.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Sat_SMHM_Host = np.load("./DataOutput/RunParam_{}/Sat_SMHM_Sat_SMHM_Host.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return z, SatHaloMass, AvaHaloMass, Surviving_Sat_SMF_MassRange, Sat_SMHM, Sat_SMHM_Host
def LoadData_Mergers(RunParam):
    Accretion_History = np.load("./DataOutput/RunParam_{}/Mergers_Accretion_History.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load("./DataOutput/RunParam_{}/Mergers_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Mergers_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/Mergers_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Accretion_History, z, AvaHaloMass, Surviving_Sat_SMF_MassRange
def LoadData_Pair_Frac(RunParam):
    Pair_Frac = np.load("./DataOutput/RunParam_{}/Pair_Frac_Pair_Frac.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load("./DataOutput/RunParam_{}/Pair_Frac_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Pair_Frac_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/Pair_Frac_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Pair_Frac, z, AvaHaloMass, Surviving_Sat_SMF_MassRange
def LoadData_Sat_Env_Highz(RunParam):
    AvaHaloMass = np.load("/data/pg1g15/Side_Projects/Analytic_DM_Model/DataOutput/RunParam_{}/Sat_Env_Highz_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load("/data/pg1g15/Side_Projects/Analytic_DM_Model/DataOutput/RunParam_{}/Sat_Env_Highz_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AnalyticalModelFrac_ = np.load("/data/pg1g15/Side_Projects/Analytic_DM_Model/DataOutput/RunParam_{}/Sat_Env_Highz_AnalyticalModelFracHighz.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AnalyticalModelNoFrac_ = np.load("/data/pg1g15/Side_Projects/Analytic_DM_Model/DataOutput/RunParam_{}/Sat_Env_Highz_AnalyticalModelNoFracHighz.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SM_Cuts = np.load("/data/pg1g15/Side_Projects/Analytic_DM_Model/DataOutput/RunParam_{}/Sat_Env_Highz_SM_Cuts.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, z, AnalyticalModelFrac_, AnalyticalModelNoFrac_, SM_Cuts
def LoadData_Raw_Richness(RunParam):
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Raw_Richness_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load("./DataOutput/RunParam_{}/Raw_Richness_Highz_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load("./DataOutput/RunParam_{}/Raw_Richness_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_Weighting_highz = np.load("./DataOutput/RunParam_{}/Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, z, Surviving_Sat_SMF_MassRange, Surviving_Sat_SMF_Weighting_highz
def LoadData_MultiEpoch_SubHalos(RunParam):
    z = np.load("./DataOutput/RunParam_{}/MultiEpoch_SubHalos_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SatHaloMass = np.load("./DataOutput/RunParam_{}/MultiEpoch_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SurvivingSubhalos_z_z = np.load("./DataOutput/RunParam_{}/MultiEpoch_SurvivingSubhalos_z_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return z, SatHaloMass, SurvivingSubhalos_z_z
def LoadData_Pair_Frac_Halo(RunParam):
    z = np.load("./DataOutput/RunParam_{}/Pair_Frac_Halo_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Pair_Frac_Halo = np.load("./DataOutput/RunParam_{}/Pair_Frac_Halo_Pair_Frac_Halo.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Accretion_History_Halo = np.load("./DataOutput/RunParam_{}/Pair_Frac_Halo_Accretion_History_Halo.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AvaHaloMass = np.load("./DataOutput/RunParam_{}/Pair_Frac_Halo_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SatHaloMass = np.load("./DataOutput/RunParam_{}/Pair_Frac_Halo_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Pair_Frac_Halo, Accretion_History_Halo, z, AvaHaloMass, SatHaloMass
#==========================Loading Output=======================================
