import numpy as np
import matplotlib.pylab as plt
import Functions as F
from astropy.cosmology import Planck15 as Cosmo_AstroPy
from colossus.cosmology import cosmology
import colossus.halo.mass_adv as massdefs
import pickle
from colossus.lss import mass_function
from scipy.integrate import cumtrapz
from colossus.halo.mass_so import M_to_R
from halotools import empirical_models
cosmology.setCosmology("planck15")
from scipy.interpolate import interp1d,interp2d
from fast_histogram import histogram2d
import pickle
import time
from multiprocessing import Pool


Cosmo = cosmology.getCurrent()
HMF_fun = F.Make_HMF_Interp() #N Mpc^-3 h^3 dex^-1
h = Cosmo.h
h_3 = h*h*h
Lbox = 100

burst_params =\
{\
 'delay':False,\
 'burst_law': 'Hopkins',\
 'test_hopkins':False,\
 'total_time_yr': 1.e9,\
  'tburst_yr' :1.e8,\
 'use_hopkinsGas':False,\
 'integrate':True\
 } #delay is whether you account for satellite initialization at the appropriate z. Set 'True' to properly acount for it.

Override =\
{\
'M10':11.95,\
'SHMnorm10':0.032,\
'beta10':1.61,\
'gamma10':0.54,\
'M11':0.4,\
'SHMnorm11':-0.02,\
'beta11':-0.6,\
'gamma11':-0.1\
}

AbnMtch =\
{\
'Behroozi13': False,\
'Behroozi18': False,\
'B18c':False,\
'B18t':False,\
'G18':True,\
'G18_notSE':False,\
'Lorenzo18':False,\
'Moster': False,\
'z_Evo':True,\
'Scatter': 0.11,\
'Override_0': False,\
'Override_z': False,\
'Override': Override,\
 'PFT':False,\
'PFT1': False,\
'PFT2': False,\
'PFT3': False\
}

Paramaters = \
{\
'AbnMtch' : AbnMtch,\
'AltDynamicalTime': 1,\
'NormRnd': 0.5,\
 'SFR_Model': 'S15'\
}

SFRstep =0.01  
SFRlog = np.arange(-8,5,SFRstep)
SFR= 10**SFRlog

#burst_law_fun = pickle.load(open('burst_law.pkl','rb'))  #arguments: log10(time[yr]), log10(Mburst_ini), returns: log10SFR(t)

    #prepare output
max_stars = 12.5; min_stars = 9; bins_stars=0.1
max_gas = 12; min_gas=6; bins_gas=0.1
max_sfr = 5 ; min_sfr = -8; bins_sfr =0.05
max_ssfr = 5 ;min_ssfr = -15; bins_ssfr =0.05
max_ssfr_MS = 2.5; min_ssfr_MS = -3; bins_ssfr_MS = 0.05
max_mu = 0.; min_mu =-3; bins_mu = 0.01

mstar_bins = np.arange(min_stars,max_stars,bins_stars)
mgas_bins = np.arange(min_gas,max_gas,bins_gas)
sfr_bins = np.arange(min_sfr,max_sfr,bins_sfr)
ssfr_bins = np.arange(min_ssfr,max_ssfr,bins_ssfr)
ssfr_bins_MS = np.arange(min_ssfr_MS,max_ssfr_MS,bins_ssfr_MS)
mu_bins = np.arange(min_mu,max_mu,bins_mu)
                          
    
mstar_len = len(mstar_bins)
mgas_len =len(mgas_bins)
sfr_len = len(sfr_bins)
ssfr_len = len(ssfr_bins)
    


def FakMa10_MergerRate_dlog10(M, zz,e, dynfrict=False):
    
    ''' M in input must be in h units'''
    
    MergerRate = np.zeros((len(M),len(e)))
    z_infall = np.zeros(len(e))
                          
    A = 0.0104; e_bar = 9.72*(10**-3);alpha = 0.133;beta=-1.995;gamma=0.263;nu=0.0993
    masses_FakMa = M - np.log10(h)
                              
    if not dynfrict:           
        
        for j,m in enumerate(M):
            for k,ee in enumerate(e):
                FakMa10 = A*np.power(10**(masses_FakMa[j]-12),alpha)*np.power(ee,beta)*np.exp(np.power((ee/e_bar),gamma))*np.power(1+zz, nu)*ee*np.log(10)
                Weight_cen = HMF_fun(m,zz)/10**m/np.log(10)
                MergerRate[j][k] = FakMa10*Weight_cen
                          
    elif dynfrict: #Jiang et al. 2008
        
        for j,m in enumerate(M):
            for k,ee in enumerate(e):
                #to fit Jiang+08 as in Shen09
                frac=(1.+0.22*(ee*np.log(1.+1./ee))**(-1.))**(2./3)
                z_infall[k]=zz*frac
                FakMa10 = A*np.power(10**(masses_FakMa[j]-12),alpha)*np.power(ee,beta)*np.exp(np.power((ee/e_bar),gamma))*np.power(1+z_infall[k], nu)*ee*np.log(10)

                Weight_cen = HMF_fun(m,z_infall[k])/10**m/np.log(10)
                MergerRate[j][k] = FakMa10*frac*Weight_cen

    return MergerRate,z_infall #merger rate per unit dz, dM dlog10e, volume


def make_burst(Mcen_star,Mcen_gas,Mcen_DM,Msat_star,Msat_gas,Msat_DM,burst_params ):
    #burst law goes here -> get Mburst
    Mcen_star = 10**Mcen_star
    Mcen_gas = 10**Mcen_gas
    Msat_gas = 10**Msat_gas
    Msat_star = 10**Msat_star
    Mbar_cen = Mcen_star + Mcen_gas #+ Mcen_DM
    Mbar_sat = Msat_star + Msat_gas #+ Msa_DM
    merger_ratio_bar = Mbar_sat/Mbar_cen
    
    # add MR >0.1 to avoid eccessive computing time
    mask_mergers = np.ma.masked_greater(merger_ratio_bar,0.1).mask
    
    Mgas_tot = Msat_gas + Mcen_gas
    Mstar_tot = Msat_star + Mcen_star
    fgas_tot = (Mgas_tot)/(Mgas_tot+Mstar_tot)
    fgas_cen = Mcen_gas/(Mcen_gas+Mcen_star)
    if burst_params['burst_law'] =='Hopkins':
        
        Mburst = Mgas_tot[mask_mergers]*merger_ratio_bar[mask_mergers]*(1-fgas_tot[mask_mergers]) # powers the high mass end of the Mburst function
       # Mburst = Mcen_gas[mask_mergers]*merger_ratio_bar[mask_mergers]*(1.-fgas_cen[mask_mergers]) #original Hopkins recipe
    else:
        #new burst law goes here    
       # Mburst = Mcen_gas[mask_mergers]*merger_ratio_bar[mask_mergers]
        Mburst = Mgas_tot[mask_mergers]*merger_ratio_bar[mask_mergers]
        pass
    
    #Mburst = 10**np.random.normal(np.log10(Mburst),0.35)  #scatter from Hopkins
    return Mburst,  mask_mergers, merger_ratio_bar #add ~0.2 dex scatter in Mburst?

#def burst_evol(Mburst,tburst,Ntimes):
    
    
#    times = 10**np.linspace(6,tburst,Ntimes)
#    dt = times[1:]-times[:-1]
#    SFR_burst = 10**burst_law_fun(np.log10(times[1:]), np.log10(Mburst))
           
#    stars_formed =  SFR_burst*dt
#    cumul_stars_formed = [np.cumsum(stars_formed[i,:]) for i in range(len(Mburst))] #CAMBIA!!!! CYTHON?
    
     
#    return SFR_burst, np.array(cumul_stars_formed)

def burst_history_log(x,t0,M0):
    mtilde = M0/t0
    return t0*np.log(10)*np.e**(-x/mtilde)

def burst_evol(Mburst,preburst_stars,burst_params,SFR,SFRstep,array_times_out, cosmo_z_out):

    #Hopkins+10 burst law
    preburst_stars = 10**preburst_stars
    t = 10**np.random.normal( np.log10(burst_params['tburst_yr']),0.1,len(Mburst)) #scatter of 0.1 dex to be discussed / tburst_yr must be in linear units!
    s = np.tile(SFR,(len(Mburst),1))
    dtdlogmdot = np.array(list(map(lambda x,y,z: burst_history_log(x,y,z), *[s,t,Mburst])))
    
    
    #SFR evolution
    res =np.cumsum(SFRstep*dtdlogmdot,axis=1)
    times = np.max(res,axis=1).reshape((len(Mburst),1)) -res

    #mass evolution
    dt = times[:,:-1]-times[:,1:]
    dtnew = np.insert(dt, 0, dt[:,0],axis=1)
    mass = np.cumsum(s*dtnew,axis=1)
    mass = np.max(mass,axis=1).reshape((len(Mburst),1))-mass

    tt = np.fliplr(times)
    ii = np.tile(array_times_out,(len(Mburst),1)) 
    indices = np.array(list(map( np.searchsorted, *[tt,ii])))
    indices[indices==times.shape[1]] = times.shape[1]-1
    
    ss =np.fliplr(s)
    SFR_out = np.array(list(map( lambda x,i: x[i], *[ss,indices])))
    mm = np.fliplr(mass)
    mstar_out =np.array(list(map (lambda x, i : x[i], *[mm,indices])))
    SFH = (preburst_stars + mstar_out.T).T
    

    
    sSFR_out = SFR_out/SFH
    
    
    #cosmo_z_out = np.tile(cosmo_z_out,(mstar_out.shape[0],1))
    #need to include gas consumption
    SFR_MS_mean = \
           np.array(list(map(\
                        lambda x,y: F.StarFormationRate(x,y, Paramaters['SFR_Model'], ScatterOn=False),\
                          *[np.log10(SFH.T),cosmo_z_out])))
    
    SFR_MS_mean = 10**SFR_MS_mean.T
    
    #sSFR_MS_mean = 10**SFR_MS_mean/SFH
    
    sSFR_MS_burst = SFR_out/SFR_MS_mean
    
    
    return SFR_out, mstar_out, sSFR_out, sSFR_MS_burst   # linear units


def hopkinsGas(mstar,z):
    #f0 = 1./ ( 1. + 10**(mstar-9.15)**0.4  )
    f0 = 1./(1 + (10**mstar/10**9.15)**0.4)
    frac = Cosmo.lookbackTime(z)/Cosmo.lookbackTime(z=500)
    
    try:
        f = f0 *(1. -frac*( 1.- f0**1.5))**(-2./3)
    except:
        f0  = np.tile(f0, (len(z),1))
        f = np.array(list(map( lambda x,y:  x*(1-y*(1.-x**1.5))**(-2./3), *[f0,frac]   )))
        
    

    mgas_hop = np.log10 ( 10**mstar*f/(1-f))

    return np.random.normal(mgas_hop,0.2)




def make_final_mstar(mstar,mask):
    
    int_mask = mask.astype(int)
    zz = np.count_nonzero(int_mask)
    mstar[mask] = mstar[zz]
            
    return mstar
        
    
def make_dndlogmdot(dndt,dtdlogmdot):
    
    dndt = np.tile(dndt, (dtdlogmdot.shape[0],1))
    res = dndt*dtdlogmdot
    
    return res


def run(z,zmin=None):#, burst_params, Lbox):
    
    tb = burst_params['total_time_yr']
        
    if burst_params['integrate']:
        cosmo_z_out = zmin
        age_univ_out = Cosmo.lookbackTime(cosmo_z_out)
        array_times_out = [(Cosmo.lookbackTime(z)-age_univ_out)*1.e9]
    else:
        array_times_out = np.array([tb*0.01,tb*0.05,tb*0.1,tb*0.15,tb*0.18,tb*0.21,tb*0.25,tb*0.5,tb*0.75])#,tb])     
        cosmo_times_out = Cosmo.lookbackTime(z)-array_times_out/1.e9
        cosmo_z_out = Cosmo.lookbackTime(cosmo_times_out, inverse=True)
#index_times_out = np.append([0], np.searchsorted(times, array_times_out))#timestep where to save output
    SFR_out = np.zeros( ( len(array_times_out)+1, len(mstar_bins)-1, len(sfr_bins)-1  ) )
    sSFR_out = np.zeros( ( len(array_times_out)+1, len(mstar_bins)-1, len(ssfr_bins)-1  ) )
    sSFR_MS_out = np.zeros( ( len(array_times_out)+1, len(mstar_bins)-1, len(ssfr_bins_MS)-1  ) )

    gas_out = np.zeros( ( len(array_times_out)+1, len(mstar_bins)-1, len(mgas_bins) -1 ) )

    start = time.time()
    max_mu = 0; min_mu =-3; bins_mu = 0.1
    mu_bins = np.arange(min_mu,max_mu,bins_mu)
  #  mu_out = np.zeros(len(mu_bins)-1)  # add also mstar dependence
    mu_out = np.zeros( (len(mstar_bins)-1,len(mu_bins)-1))  # add also mstar dependence

    #################################################################################
    ###################################### Main starts here #########################
    #################################################################################
    
    #read and make merger rate
    Vol = (Lbox/h)**3
    
    halos_in=  np.arange(11,16,0.1) #in h units as required by HMF and FakMa
   # z_in = np.loadtxt('zSTEEL.txt')
    ebin=0.01
    elog = np.arange(-2.,0,ebin)
    e = 10**elog
    
    #FakMa10 requires M in h units
    MRlog, z_inf = FakMa10_MergerRate_dlog10(halos_in,z,e,dynfrict=True) #merger rate puntuale
    MRlog = MRlog*h_3  
    
    halos_in = halos_in -np.log10(h) #now you can de-h the masses
    mergerRate = interp2d(e,halos_in,MRlog)
    z_infall = interp1d(elog,z_inf)
    
    #make mock of centrals
    haloMF = HMF_fun(halos_in,z)
    
    step = halos_in[1]-halos_in[0]
    Ncum=Vol*(np.cumsum((haloMF*h_3*step)[::-1])[::-1])
    f = interp1d(Ncum,halos_in)
    array_cumul=np.arange(min(Ncum),max(Ncum))
    halos=f(array_cumul)[::-1]
    
    mstar_cen = F.DarkMatterToStellarMass(halos, z, Paramaters, ScatterOn=True)
    SFR_cen = F.StarFormationRate(mstar_cen,z, Paramaters['SFR_Model'])
    
    if not burst_params['use_hopkinsGas']:
        mgas_cen = F.GetGasMass(mstar_cen, z, halos,Paramaters['SFR_Model'])

    # use hopkins for test
    else:
        mgas_cen =hopkinsGas(mstar_cen, z)
    
    #assign merger rate to galaxies in a given halo bin
    halowidth = 0.2
    halobins = np.arange(11,16,halowidth)
    centers = halobins[1:]-halowidth/2.
    
    for i,halocen in enumerate(centers):   # cambia il for! cython? spacchetta in multiprocessing?
        print('halocenter:' +str(halocen))
        #here I compute satellite accretion for centrals grouped in halo bins to avoid using all the central halos
        elog_bins = elog[1]-elog[0]
        cumul_MRlog = np.cumsum(elog_bins*mergerRate(e,halocen))
        cumul_interp = interp1d(cumul_MRlog,elog)
        array_cumul = np.random.uniform(min(cumul_MRlog),max(cumul_MRlog),size=len(mstar_cen))
        e_out = cumul_interp(array_cumul)
        halosat = halocen+e_out # product in log space
        
        if burst_params['delay']:
            
            #compute dynamical friction and go backwards in time to z_infall (which actually is an upper limit!)   --> Use instead Shen2009. z_infall is provided in eq. 9,10,11
            
           # tdyn = F.DynamicalFriction(halocen, halosat, z, Paramaters) #Gyr
           # age_now = Cosmo.age(z)
           # age_infall = age_now -tdyn #uyninverse was younger
           # age_universe = Cosmo.age(z=0) 
           # lookback = age_universe-age_infall #if I put age_now rather than age_infall at the next step I would get the present z again by definition, instead the infall was at higher z 
           # z_infall = Cosmo.lookbackTime(lookback, inverse=True)
            
            # 
            these_z_infall = z_infall(e_out)
            mstar_sat = F.DarkMatterToStellarMass(halosat, these_z_infall, Paramaters, ScatterOn=True)
            if not burst_params['use_hopkinsGas']:
                mgas_sat = F.GetGasMass(mstar_sat,these_z_infall,halosat, Paramaters['SFR_Model'])
            
            #use hopkins to test
            else:
                mgas_sat = hopkinsGas(mstar_sat, z)
            
            
        else:
            mstar_sat = F.DarkMatterToStellarMass(halosat, z, Paramaters, ScatterOn =True)
            mgas_sat = F.GetGasMass(mstar_sat,z,halosat, Paramaters['SFR_Model'])
        #these_z_infall = z_infall(e_out)
        #if not burst_params['use_hopkinsGas']:
            #mgas_sat = F.GetGasMass(mstar_sat,z,halosat, Paramaters['SFR_Model'])
            
            #use hopkins to test
        #else:
        #    mgas_sat = hopkinsGas(mstar_sat, z)    
    
        #assign satellites to centrals in that halo bin
    
        binning = np.ma.masked_inside(halos, halocen-halowidth/2, halocen+halowidth/2).mask
        
        Mburst, mask_mergers, merger_ratio_bar = make_burst(mstar_cen[binning],mgas_cen[binning],halocen,mstar_sat[binning],mgas_sat[binning],halosat[binning],burst_params)
        

        if len(Mburst) > 1:
            preburst_stars = np.log10(10**mstar_cen[binning][mask_mergers] + 10**mstar_sat[binning][mask_mergers])            
            SFR_burst, stars_formed_burst, sSFR_burst, sSFR_MS_burst = burst_evol(Mburst,preburst_stars,burst_params,SFR,SFRstep,array_times_out,cosmo_z_out)
            
            
            SFH = (preburst_stars + stars_formed_burst.T).T
            premerg_stars = 10**mstar_cen[binning][mask_mergers]
          #  SFH =  np.insert(SFH,0,premerg_stars, axis=1) # all the lightcurves since the initial mass/SFR
        
            SFR_preburst = 10**SFR_cen[binning][mask_mergers]
           # SFR_burst =  np.insert(SFR_burst,0,SFR_preburst, axis=1) #NB the assumption here is that the SFR of the burst is INDEPENDENT on the preburst!! That is, the burst is not additive with the MS SFR. In the Fensch+ scenario instead the burst would be MULTIPLICATIVE wrt the MS SFR, because everything is driven by the increase in gas turbulence and shocks wrt the pre-burst condition.

        
            mgas_preburst = 10**mgas_cen[binning][mask_mergers] + 10**mgas_sat[binning][mask_mergers]
            mgas_new =  (mgas_preburst- stars_formed_burst.T).T
            mgas_premerger= 10**mgas_cen[binning][mask_mergers]
          #  mgas_new = np.insert(mgas_new,0,mgas_premerger, axis=1)
          #  sSFR_preburst = 10**SFR_cen[binning][mask_mergers]/10**mstar_cen[binning][mask_mergers]
          #  sSFR_burst =  np.insert(sSFR_burst,0,sSFR_preburst, axis=1)
            
            if any(mgas_new.flatten() == 0 ):
        #sets asymptotic mstar value and low SFR if mgas gets below 0
                mask_gas = np.ma.masked_less(mgas_new,0).mask #returns True where <0
                SFR_burst[mask_gas] = min(SFR)
                SFH = np.array(list(map( lambda x,y : make_final_mstar(x,y), *[SFH,mask_gas]))) 
        
            SFR_burst = np.log10(SFR_burst)
            SFH = np.log10(SFH)
            sSFR_burst = np.log10(sSFR_burst)
            sSFR_MS_burst = np.log10(sSFR_MS_burst)
            
 #           for s in range(len(array_times_out)):  #create 2d histogram at each time of interest
 #               SFR_out[s] = SFR_out[s] + histogram2d(SFH[:,s],SFR_burst[:,s], (mstar_len,sfr_len ), ((min_stars,max_stars),(min_sfr,max_sfr)) )
 #               sSFR_out[s] = sSFR_out[s] + histogram2d(SFH[:,s],sSFR_burst[:,s], (mstar_len,ssfr_len ), ((min_stars,max_stars),(min_ssfr,max_ssfr)) )
            #mstar_out[s] = mstar_out[s] + histogram1d(mstar_new[:,s]) 
 #               gas_out[s] = gas_out[s] + histogram2d(SFH[:,s],mgas_new[:,s], (mstar_len,mgas_len ), ((min_stars,max_stars),(min_gas,max_gas)) )

            merger_ratio_bar = np.log10(merger_ratio_bar)
            mu_out = mu_out + np.histogram2d(mstar_cen[binning],merger_ratio_bar,bins=(mstar_bins,mu_bins))[0]
            #print(len(mstar_cen),len(merger_ratio_bar))
            #print(np.histogram2d(mstar_cen,merger_ratio_bar,bins=(mstar_bins,mu_bins))[0])
            for s in range(len(array_times_out)):  #create 2d histogram at each time of interest
                SFR_out[s] = SFR_out[s] + np.histogram2d(SFH[:,s],SFR_burst[:,s], bins=(mstar_bins,sfr_bins) )[0]
                sSFR_out[s] = sSFR_out[s] + np.histogram2d(SFH[:,s],sSFR_burst[:,s], bins=(mstar_bins,ssfr_bins))[0]
            #mstar_out[s] = mstar_out[s] + histogram1d(mstar_new[:,s]) 
                sSFR_MS_out[s] = sSFR_MS_out[s] + np.histogram2d(SFH[:,s],sSFR_MS_burst[:,s], bins=(mstar_bins,ssfr_bins_MS))[0]

                gas_out[s] = gas_out[s] + np.histogram2d(SFH[:,s],mgas_new[:,s],bins=(mstar_bins,mgas_bins)  )[0]            
            
                                
    if burst_params['use_hopkinsGas']:
        file_SFR = './quick_output/integration/Hopkins/SFR'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))
        file_sSFR = './quick_output/integration/Hopkins/sSFR'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))
        file_sSFR_MS = './quick_output/integration/Hopkins/sSFR_MS'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))
        file_mgas = './quick_output/integration/Hopkins/mgas'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))    
        file_mu = './quick_output/integration/Hopkins/mu'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))    
    else:
        file_SFR = './quick_output/integration/sargschr/SFR'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))
        file_sSFR = './quick_output/integration/sargschr/sSFR'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))
        file_sSFR_MS = './quick_output/integration/sargschr/sSFR_MS'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))
        file_mgas = './quick_output/integration/sargschr/mgas'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z)) 
        file_mu = './quick_output/integration/sargschr/mu'.format("".join(("{}_".format(i) for i in burst_params)))+str(Paramaters['SFR_Model']+'z'+str(z))
    
    np.save(file_SFR, SFR_out)
    np.save(file_sSFR, sSFR_out)
    np.save(file_sSFR_MS, sSFR_MS_out)
    np.save(file_mgas, gas_out)
    np.save(file_mu,mu_out)
    end = time.time()
    
    print('For a box of '+str(Lbox)+' Mpc/h the program ran in %.3f s' %(end-start))
    return
    

    
def test_hopkins(z, burst_params, Lbox,burst_bins,mstar_low, mstar_up):
    
    start = time.time()

    cosmo_times_out = Cosmo.lookbackTime(z)-array_times_out/1.e9
    cosmo_z_out = Cosmo.lookbackTime(cosmo_times_out, inverse=True)
    #read and make merger rate
    Vol = (Lbox/h)**3
    
    halos_in=  np.arange(11,16,0.1) #in h units as required by HMF and FakMa
   # z_in = np.loadtxt('zSTEEL.txt')
    ebin=0.01
    elog = np.arange(-2.,0,ebin)
    e = 10**elog
    
    #FakMa10 requires M in h units
    MRlog, z_inf = FakMa10_MergerRate_dlog10(halos_in,z,e,dynfrict=True) #merger rate puntuale
    MRlog = MRlog*h_3  
    
    halos_in = halos_in -np.log10(h) #now you can de-h the masses
    mergerRate = interp2d(e,halos_in,MRlog)
    z_infall = interp1d(elog,z_inf)
    
    #make mock of centrals
    haloMF = HMF_fun(halos_in,z)
    
    step = halos_in[1]-halos_in[0]
    Ncum=Vol*(np.cumsum((haloMF*h_3*step)[::-1])[::-1])
    f = interp1d(Ncum,halos_in)
    array_cumul=np.arange(min(Ncum),max(Ncum))
    halos=f(array_cumul)[::-1]
    
    mstar_cen = F.DarkMatterToStellarMass(halos, z, Paramaters, ScatterOn=True)
    mgas_cen = F.GetGasMass(mstar_cen, z, halos,Paramaters['SFR_Model'])
    SFR_cen = F.StarFormationRate(mstar_cen,z, Paramaters['SFR_Model'])
    
    # use hopkins for test
    
   # mgas_cen =hopkinsGas(mstar_cen, z)
    
    #assign merger rate to galaxies in a given halo bin
    halowidth = 0.2
    halobins = np.arange(11,16,halowidth)
    centers = halobins[1:]-halowidth/2.
    
    hist_burst = np.zeros(len(burst_bins)-1)
    
    for i,halocen in enumerate(centers):   # cambia il for! cython? spacchetta in multiprocessing?
        print('halocenter:' +str(halocen))
        #here I compute satellite accretion for centrals grouped in halo bins to avoid using all the central halos
        elog_bins = elog[1]-elog[0]
        cumul_MRlog = np.cumsum(elog_bins*mergerRate(e,halocen))
        cumul_interp = interp1d(cumul_MRlog,elog)
        array_cumul = np.random.uniform(min(cumul_MRlog),max(cumul_MRlog),size=len(mstar_cen))
        e_out = cumul_interp(array_cumul)
        halosat = halocen+e_out # product in log space
        
        if burst_params['delay']:
            
            #compute dynamical friction and go backwards in time to z_infall (which actually is an upper limit!)   --> Use instead Shen2009. z_infall is provided in eq. 9,10,11
            
           # tdyn = F.DynamicalFriction(halocen, halosat, z, Paramaters) #Gyr
           # age_now = Cosmo.age(z)
           # age_infall = age_now -tdyn #uyninverse was younger
           # age_universe = Cosmo.age(z=0) 
           # lookback = age_universe-age_infall #if I put age_now rather than age_infall at the next step I would get the present z again by definition, instead the infall was at higher z 
           # z_infall = Cosmo.lookbackTime(lookback, inverse=True)
            
            # 
            these_z_infall = z_infall(e_out)
            mstar_sat = F.DarkMatterToStellarMass(halosat, these_z_infall, Paramaters, ScatterOn=True)
            mgas_sat = F.GetGasMass(mstar_sat,these_z_infall,halosat, Paramaters['SFR_Model'])
            
            #use hopkins to test
            
            #mgas_sat = hopkinsGas(mstar_sat, these_z_infall)
            
            
        else:
            mstar_sat = F.DarkMatterToStellarMass(halosat, z, Paramaters, ScatterOn =True)
            mgas_sat = F.GetGasMass(mstar_sat,z,halosat, Paramaters['SFR_Model'])    
    
    
        #assign satellites to centrals in that halo bin
    
        binning = np.ma.masked_inside(halos, halocen-halowidth/2, halocen+halowidth/2).mask
        mstar_binning = np.ma.masked_inside(mstar_cen[binning],mstar_low,mstar_up).mask
        
        Mburst, _ = make_burst(mstar_cen[binning][mstar_binning],mgas_cen[binning][mstar_binning],halocen,mstar_sat[binning][mstar_binning],mgas_sat[binning][mstar_binning],halosat[binning][mstar_binning],burst_params)
        
        
        hist_burst = hist_burst + np.histogram(np.log10(Mburst), bins=burst_bins)[0]
        
        
        
    return hist_burst
        

if __name__=='__main__':
    #z=2.5
    #Lbox=150
    
    
    
    if burst_params['test_hopkins']:
        mstar_low =10.5
        mstar_up =10.8
        z2 = Cosmo.lookbackTime(z=2)#*1.e9
        z3 = Cosmo.lookbackTime(z=2.5)#*1.e9
        dt = 0.1
        t = np.arange(z2,z3,dt)
        t_yr = t*1.e9
        dt_yr =dt*1.e9
        zs = Cosmo.lookbackTime(t, inverse=True)[::-1]   # z descends: first compute high z, then low. This reflects in gradient.append
        
        burst_bin_width =0.1
        burst_bins =np.arange(8,12.3,burst_bin_width)
        hist_burst = np.zeros((len(t),len(burst_bins)-1))       
        
        for i in range(len(t)):
            hist_burst[i] = test_hopkins(zs[i], burst_params, Lbox, burst_bins,mstar_low,mstar_up)  #bottleneck
            
           
        
        gradient = []
        for i in range(len(t)-1):
            gradient.append( ( hist_burst[i+1,:]-hist_burst[i,:])/dt_yr)   # (N(mburst,t_{i+1}) - N(mburst, t_i) )/dt
        
        gradient = np.array(gradient)
        ss = np.tile(10**sfr_bins, (len(burst_bins)-1,1) )
        bb =10**(burst_bins[1:]-burst_bin_width/2.)  
        times = 10**np.random.normal(np.log10(burst_params['tburst_yr']),0.1, size=len(burst_bins))
        burst_law = np.array(list(map( lambda x,y,z:  burst_history_log(x,z, y), *[ss,bb,times]))) # each row is the dtdlogmdot of a mburst
        
        burst_law = burst_law.T #needed to multiply with gradient
        burst_law = np.tile(burst_law, (gradient.shape[0],1,1))
        print(burst_law.shape)
        dndlogmdot  = np.array(list(map( lambda x,y: make_dndlogmdot(x,y),  *[gradient, burst_law] )))
        
        filesav = './quick_output/dndlogmdot_'+str(mstar_low)+'_'+str(mstar_up)
        filesav_dt = './quick_output/dndt_'+str(mstar_low)+'_'+str(mstar_up)
        
        np.save(filesav, dndlogmdot)
        np.save(filesav_dt,gradient)
    else:
        
        #z = [2.1,2.2,2.3,2.4]
        #with Pool(5) as p:
        #    p.map(run,z)
        if burst_params['integrate']:
            redshifts=np.arange(2,2.55,0.05)[::-1]
            for z in redshifts:
                print(round(z,2))
                z = round(z,2)
                run(z,zmin=[2.,2])
    #call function
    
    
