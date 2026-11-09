import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial

import scipy.integrate as integ
from scipy import interpolate
from quadax import cumulative_trapezoid

from ripple.waveforms import TaylorF2
from ripple import ms_to_Mc_eta

# import sys
# sys.path.append("<path to a local lalsuite install>")
# from lalsimulation import SimInspiralTaylorF2

from bns_tides import run_parameters as rp


G, c, Ms = 6.67259e-08, 29979245800.0, 1.989e+33


@jit
def omega_UR(x):
    """ Calculate omega from ln(dim'less lambda) using universal relations
    Chan et. al. 2015 https://arxiv.org/pdf/1408.3789
    """
    a0 = 1.820e-1
    a1 = -6.836e-3
    a2 = -4.196e-3
    a3 = 5.215e-4
    a4 = -1.857e-5
    
    return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

@jit
def inertia_UR(x):
    """ Calculate the moment of inertia from ln(dim'less lambda) using universal relations
    Yagi Yunes 2017 https://doi.org/10.1016/j.physrep.2017.03.002
    """
    ai = 1.496
    bi = 5.951e-2
    ci = 2.238e-2
    di = -6.953e-4
    ei = 8.345e-6
    
    return jnp.exp(ai + bi*x + ci*x**2 + di*x**3 + ei*x**4)


def aLIGOSn(f, f0=215, Sm=1.0e-49):
    """
    """
    # assert np.min(f) > 9.9, 'frequencies below Sn domain: 10 Hz'
    # assert np.max(f) < 2000.1, 'frequencies above Sn domain, 2000 Hz'
    fof0 = f/f0
    term3 = (1 - fof0**2 + 0.5*fof0**4) / (1 + 0.5*fof0**2)
    Sn = Sm*(fof0**(-4.14) - 5*fof0**(-2) + 111*term3)
    
    return Sn


def CE_Sn(f):
    """ Cosmic Explorer Strain Curve
    https://dcc.cosmicexplorer.org/CE-T2000017/public
    https://dcc.cosmicexplorer.org/public/0163/T2000017/008/ce_strain.pdf
    """
    # Path scrubbed for this repo: the detector curves are not included here, and the story never calls CE_Sn.
    ce_strain_file = "detector_strain_curves/ce_strain/cosmic_explorer_strain.txt"
    # ce_strain_file = "..\\detector_strain_curves\\ce_strain\\cosmic_explorer_strain.txt"
    
    # Read in strain data, splitting lines and dropping "\n"
    with open(ce_strain_file, 'r') as f:
        ce_strain_data = f.read().splitlines()
    
    # Split x,y coords by ' ' in string, convert str to float in numpy array
    CE_Sn_arr = np.array([line.split(' ') for line in ce_strain_data], dtype=float)
    print(type(CE_Sn_arr[5,0]))
    
    # Create interpolating function from CE data
    CE_Sn_interp = interpolate.interp1d(CE_Sn_arr[:,0], CE_Sn_arr[:,1])
    print(type(CE_Sn_arr[5,0]))
    # Evaluate CE interp function at input frequencies
    CE_Sn_f = CE_Sn_interp(f)
    
    return CE_Sn_f


def inner(h1, h2, PSD, freq):
    """
    """
    return integ.trapezoid((2*jnp.real(np.conjugate(h1)*h2))/PSD, freq)


def waveform_mismatch(h1, h2, PSD, freq, f_upper):
    """
    """
    mask = freq <= f_upper
    h1m, h2m, PSDm, freqm = h1[mask], h2[mask], PSD[mask], freq[mask]
    
    inner12 = inner(h1m, h2m, PSDm, freqm)
    inner11 = inner(h1m, h1m, PSDm, freqm)
    inner22 = inner(h2m, h2m, PSDm, freqm)
    return 1 - inner12/np.sqrt(inner11*inner22)


def get_psi_pp_newt(
    freq, # Hz
    Mchirp, # Grams
    q,
    tc, # sec
    phic # rad
):
    """
    """
    tMch = G*Mchirp/c**3;
    mu_on_Mtot = q/((1+q)**2);
    qMtot = q**(-3/5) * (1+q)**(6/5);
    
    psi_newt = 2 * np.pi * freq * tc - phic - np.pi/4  \
               + 3/4 * (8 * np.pi * tMch * freq)**(-5/3)  \
               * (1 + 20/9 * (743/336 + 11/4 * mu_on_Mtot)  \
               * (np.pi * freq * tMch * qMtot)**(2/3)  \
               - 16 * np.pi * (np.pi * freq * tMch * qMtot) );
    
    return psi_newt
                

@jit
def get_psi_pp_taylorF2_ripple(
    freq, # Hz GW frequency vector; must be linearly spaced
    # Mchirp, # Msolar
    # q, # Mass ratio M1/M2
    M1, # Solar Mass of companion 1
    M2, # Solar Mass of companion 2
    # t_coal, # sec Coalescence time
    # phi_coal, # rad Coalescence phase
    chi1, # z component of spin of companion 1
    chi2, # z component of spin of companion 2
):
    """
    Requires linear freq vector
    :params: [Mchirp, eta, chi1, chi2, Dist, t_coal, phi_coal]
    """
    cm_in_Mpc = 3.240779e-25 # Mpc; 1 cm = this many Mpc
    
    # if rp.classify_vector_type(freq) != 'linear':
    #     raise ValueError("Frequency vector can only be linear spacing.")
    
    # deltaF = float(round(freq[1]-freq[0], 12))
    # deltaF = round(freq[1]-freq[0], 12)
    # Mch, eta = ms_to_Mc_eta(jnp.array([M1, M2]))
    # params = jnp.array([Mch, eta, chi1, chi2, 0, 0, 1, t_coal, phi_coal])
    # params = jnp.array([Mch, eta, chi1, chi2, 0, 0, 1, 0, 0])
    # inspiral = TaylorF2.gen_TaylorF2(freq, params, f_ref=0, use_lambda_tildes=False)
    int_params = jnp.array([M2, M1, chi2, chi1, 0, 0])
    ext_params = jnp.array([1*cm_in_Mpc, 0, 0])
    inspiral = TaylorF2._gen_TaylorF2(freq, int_params, ext_params, f_ref=0)
    # indStart = round(freq[0]/(freq[1]-freq[0]))
    
    psi = jnp.unwrap(jnp.angle(jnp.conjugate(inspiral)))

    # return psi
    return psi # + (2 * np.pi * freq * t_coal) - phi_coal


@jit
def get_h_pp_taylorF2_ripple(
    freq, # Hz GW frequency vector; must be linearly spaced
    # Mchirp, # Msolar
    # q, # Mass ratio M1/M2
    M1, # Solar Mass of companion 1
    M2, # Solar Mass of companion 2
    # t_coal, # sec Coalescence time
    # phi_coal, # rad Coalescence phase
    chi1, # z component of spin of companion 1
    chi2, # z component of spin of companion 2
):
    """
    Requires linear freq vector
    
    :int_params: [M2, M1, chi2, chi1, tidal deform2, tidal deform1]
    ** Ripple package uses M1 as the larger and M2 as the smaller object **
    ** So I've flipped 1 and 2 here **

    :ext_params: [dist_Mpc, t_coal, phi_coal]
    """
    cm_in_Mpc = 3.240779e-25 # Mpc; 1 cm = this many Mpc
    
    # deltaF = float(round(freq[1]-freq[0], 12))
    # deltaF = round(freq[1]-freq[0], 12).astype(np.float64)
    # Mch, eta = ms_to_Mc_eta(jnp.array([M1, M2]))
    # params = jnp.array([Mch, eta, chi1, chi2, 0, 0, 1*cm_in_Mpc, 0, 0])
    # inspiral = TaylorF2.gen_TaylorF2(freq, params, f_ref=0, use_lambda_tildes=False)
    int_params = jnp.array([M2, M1, chi2, chi1, 0, 0])
    ext_params = jnp.array([1*cm_in_Mpc, 0, 0])
    inspiral = TaylorF2._gen_TaylorF2(freq, int_params, ext_params, f_ref=0)

    return jnp.conjugate(inspiral)


# def get_psi_pp_taylorF2_LAL(
#     freq, # Hz GW frequency vector; must be linearly spaced
#     # Mchirp, # Msolar
#     # q, # Mass ratio M1/M2
#     m1_SI, # kg Mass of companion 1
#     m2_SI, # kg Mass of companion 2
#     t_coal, # sec Coalescence time
#     phi_coal, # rad Coalescence phase
#     S1z, # z component of spin of companion 1
#     S2z, # z component of spin of companion 2
# ):
#     """
#     Requires linear freq vector
#     """
#     # if rp.classify_vector_type(freq) != 'linear':
#     #     raise ValueError("Frequency vector can only be linear spacing.")
    
#     # deltaF = float(round(freq[1]-freq[0], 12))
#     deltaF = round(freq[1]-freq[0], 12)
#     # print('deltaF: ', type(deltaF))
#     # print('M1: ', type(m1_SI))
#     # print('M2: ', type(m2_SI))
#     inspiral = SimInspiralTaylorF2(
#         phi_ref=0, # rad reference orbital phase
#         deltaF=deltaF, # Hz freq resolution
#         m1_SI=m1_SI, # kg
#         m2_SI=m2_SI, # kg
#         S1z=S1z, # Dimensionless z component of spin
#         S2z=S2z, # Dimensionless z component of spin
#         fStart=freq[0], # Hz Starting GW freq
#         fEnd=freq[-1]+deltaF, # Hz Highest GW freq; ends at Schw. ISCO
#         f_ref=0, # Reference GW freq; 0=coalescence
#         r=1, # m Distance to source sets strain amplitude
#         LALpars=None
#     )
    
#     indStart = round(freq[0]/(freq[1]-freq[0]))
    
#     psi = np.unwrap(np.angle(np.conjugate(inspiral.data.data[indStart:])))

#     return psi + (2 * np.pi * freq * t_coal) - phi_coal

@jit
def get_ba_l2(
    ff,
    M1,
    M2,
    R1, 
    lam_M1_5,
    wa01, 
    A_nl_1 = -3e-2,
    w_nl_ref = np.pi * 1000,
    OmegaS1=0, 
    gr_flag=1
):
    """
    all inputs are in cgs except lam_M1_5 and lam_M2_5
    """
    n_pt = len(ff)
    
    Mt = M1 + M2
    mu = M1 * M2/Mt
    
    eta = mu / Mt
    X1, X2 = M1/Mt, M2/Mt
    
    ww = np.pi * ff
    rr = jnp.cbrt(G*Mt/ww**2.)

    tMt = G*Mt/c**3

    # eqs. 66 & 67 of 2404.00147
    vv = jnp.cbrt(tMt * ww)
    xx = vv * vv
    dwdt = 96./5 * eta * vv**11. / tMt**2.
    t_gw = ww/dwdt
    dlogrdt = - 2./3. / t_gw

    # eqs. 4.7 & 4.8 of S+21
    # zz1 = 1. - 0.5 * X2 * (3.-X1) * xx * gr_flag
    # omega_FD_SL1 = 0.5 * X2 * (3.+X1) * xx * ww * gr_flag
    zz1 =     1. + ( xx/2. * X2 * (X1 - 3.)  \
                    +  xx**2/24. * X2 *(5*X1*X1*X2 -  9*X1 -  6*X1*X2 - 27) ) * gr_flag
    omega_FD_SL1 = ( xx/2. * X2 * (X1 + 3.)  \
                    -  xx**2/24. * X2 *(  X1*X1*X2 - 45*X1 + 30*X1*X2 - 27) ) * gr_flag * ww 

    # modes in M1
    ll1 = jnp.array([2, 2, 2])
    mm1 = jnp.array([2, 0, -2])
    # co-rot frame freq
    wa1 = - mm1/ll1 * OmegaS1 + wa01

    # eq. 9 of 2404.00147 less phase
    Wlm1 = jnp.array([jnp.sqrt(3.*np.pi/10), -jnp.sqrt(np.pi/5), jnp.sqrt(3.*np.pi/10)])
    rM1 = G*M1/c**2.
    k2 = 1.5 * lam_M1_5 * (rM1/R1)**5
    Qa = jnp.sqrt(k2 * 5/4./np.pi)
    va1 = jnp.zeros((3, n_pt))
    
    # va1[0, :] = (M2/M1) * (R1/rr)**(ll1[0]+1.) * Wlm1[0] * Qa
    # va1[1, :] = (M2/M1) * (R1/rr)**(ll1[0]+1.) * Wlm1[1] * Qa
    # va1[2, :] = (M2/M1) * (R1/rr)**(ll1[0]+1.) * Wlm1[2] * Qa
    va1 = va1.at[0].set((M2/M1) * (R1/rr)**(ll1[0]+1.) * Wlm1[0] * Qa)
    va1 = va1.at[1].set((M2/M1) * (R1/rr)**(ll1[0]+1.) * Wlm1[1] * Qa)
    va1 = va1.at[2].set((M2/M1) * (R1/rr)**(ll1[0]+1.) * Wlm1[2] * Qa)
    
    # eq. 18 of 2404.00147 including GR & NL tide
    Del1 = jnp.zeros((3, n_pt))
    ba_eq1 = jnp.zeros((3, n_pt), dtype=np.complex64)

    for i in range(3):
        # technically this is not true for the m=0  mode; but it is off resonance so should be fine
        # Del1[i, :] = zz1 * (wa1[i] + mm1[i] * OmegaS1) - mm1[i] * ww + mm1[i] * omega_FD_SL1 \
        #     + A_nl_1 * (ww/w_nl_ref)**2. * wa01
        Del1 = Del1.at[i].set( zz1 * (wa1[i] + mm1[i] * OmegaS1) - mm1[i] * ww + mm1[i] * omega_FD_SL1 \
            + A_nl_1 * (ww/w_nl_ref)**2. * wa01 )
        
        # ba_eq1[i, :] = zz1 * Del1[i, :] * wa01 * va1[i, :] \
        #     / (Del1[i, :]**2. - 1j * (mm1[i] *dwdt - Del1[i, :]*(ll1[i] + 1.) * dlogrdt))
        ba_eq1 = ba_eq1.at[i].set( zz1 * Del1[i, :] * wa01 * va1[i, :] \
            / (Del1[i, :]**2. - 1j * (mm1[i] *dwdt - Del1[i, :]*(ll1[i] + 1.) * dlogrdt)) )

    # Newtonian, linear resonance 
    wres_N1 = wa1[0] /2. + OmegaS1
    
    vv_res = jnp.cbrt(tMt * wres_N1)
    xx_res = vv_res * vv_res
    
    # corrected resonance; GR makes it higher, NL makes it lower
    wres1 = (1 + eta * xx_res * gr_flag + A_nl_1 * (wres_N1/w_nl_ref)**2.) * wres_N1 \
    # wres1 = (1 + eta * xx_res * gr_flag) * (wres_N1 + A_nl_1 * (wres_N1/w_nl_ref)**2. *wa01/2)

    vv_res = jnp.cbrt(tMt * wres1)
    xx_res = vv_res * vv_res    

    # eqs. 25-26 of 2404.00147
    dwdt_r = 96./5 * eta * vv_res**11. / tMt**2.
    tau_r = 5./256. / eta * tMt * (1./vv_res**8. - 1./vv**8.)
    phi_r = 1./32./eta * (1./vv_res**5. - 1./vv**5.)
    
    uu = jnp.sqrt(dwdt_r) * tau_r

    a1 = (2.*2.-9)/6 * jnp.sqrt(dwdt_r/wres1**2.)
    a2 = - 2./3*(2+1.) * jnp.sqrt( dwdt_r/wres1**2.)
    # a3 = a2 - a1/2. #-(10.*2-1)/12 * jnp.sqrt(( dwdt_r/wres1**2.)

    # eq. 24 of 2404.00147
    dFu = 2.j * (1. + a1*uu) * jnp.exp(-1j*uu**2.) / (1 + a2*uu + 2.j*uu**2.)**2.
    # Fu1 = integ.cumulative_trapezoid(dFu, x=uu, initial=0)
    Fu1 = cumulative_trapezoid(dFu, x=uu, initial=0)
    

    # eqs. 23, 30, and 31 of 2404.00147
    rr_r = jnp.cbrt(G*Mt/wres1**2.)
    va_r1 = (M2/M1) * (R1/rr_r)**(ll1[0]+1.) * Wlm1[0] * Qa
    ba_dyn = zz1 * wa01 * va_r1/jnp.sqrt(dwdt_r) * Fu1 \
            * jnp.exp(1j * (2. * (phi_r - wres1 * tau_r)))

    ba1 = ba_eq1.copy()
    # ba1[0, :] += ba_dyn
    ba1 = ba1.at[0].add( ba_dyn )

    return ba1, ba_eq1, va1, Fu1


def get_Lorentzian(
    ff,
    M1,
    M2,
    R1,
    wa01, 
    A_nl_1 = -3e-2,
    w_nl_ref = np.pi * 1000,
    OmegaS1=0, 
    gr_flag=1
):
    """
    """
    n_pt = len(ff)
    
    Mt = M1 + M2
    mu = M1 * M2/Mt
    
    eta = mu / Mt
    X1, X2 = M1/Mt, M2/Mt
    
    ww = np.pi * ff
    rr = jnp.cbrt(G*Mt/ww**2.)

    tMt = G*Mt/c**3

    # eqs. 66 & 67 of 2404.00147
    vv = jnp.cbrt(tMt * ww)
    xx = vv * vv
    dwdt = 96./5 * eta * vv**11. / tMt**2.
    t_gw = ww/dwdt
    dlogrdt = - 2./3. / t_gw

    # eqs. 4.7 & 4.8 of S+21
    zz1 = 1. - 0.5 * X2 * (3.-X1) * xx * gr_flag
    omega_FD_SL1 = 0.5 * (4.*X2 - X2**2.) * xx * ww * gr_flag

    # modes in M1
    ll1 = jnp.array([2, 2, 2])
    mm1 = jnp.array([2, 0, -2])
    # co-rot frame freq
    wa1 = - mm1/ll1 * OmegaS1 + wa01
    
    # eq. 18 of 2404.00147 including GR & NL tide
    Del1 = jnp.zeros((3, n_pt))
    LL1 = jnp.zeros((3, n_pt), dtype=np.complex64)

    for i in range(3):
        # technically this is not true for the m=0  mode; but it is off resonance so should be fine
        Del1[i, :] = zz1 * (wa1[i] + mm1[i] * OmegaS1) - mm1[i] * ww + mm1[i] * omega_FD_SL1 \
            + A_nl_1 * (ww/w_nl_ref)**2. * wa01  
        LL1[i, :] = zz1 * Del1[i, :] * wa01 \
            / (Del1[i, :]**2. - 1j * (mm1[i] *dwdt - Del1[i, :]*(ll1[i] + 1.) * dlogrdt))

    Lorentzian1 = np.real(LL1[0, :] + np.conj(LL1[-1, :]))/2.
    return Lorentzian1


@jit
def get_dpsi(
    ff,
    M1, R1, lam_M1_5, wa01, 
    M2, R2, lam_M2_5, wa02, 
    A_nl_1 = -3e-2,
    A_nl_2 = -3e-2,
    w_nl_ref = np.pi * 1000,
    OmegaS1 = 0, # rad
    OmegaS2 = 0, # rad
    gr_flag = 1
):
    
    Mt = M1 + M2
    mu = M1 * M2/Mt
    
    eta = mu / Mt
    X1, X2 = M1/Mt, M2/Mt
    
    ww = np.pi * ff
    rr = jnp.cbrt(G*Mt/ww**2.)
    g_unit = G*Mt/rr**2.
    E_orb = G*M1*M2/(-2.*rr)

    tMt = G*Mt/c**3

    E1 = G*M1**2./R1
    E2 = G*M2**2./R2

    # eqs. 66 & 67 of 2404.00147
    vv = jnp.cbrt(tMt * ww)
    xx = vv * vv
    dwdt = 96./5 * eta * vv**11. / tMt**2.
    t_gw = ww/dwdt
    dlogrdt = - 2./3. / t_gw  
    dEdt_pp = 2./3. * E_orb/t_gw

    # modes in M1
    ll1 = jnp.array([2, 2, 2])
    mm1 = jnp.array([2, 0, -2])
    # co-rot frame freq
    wa1 = - mm1/ll1 * OmegaS1 + wa01
    
    Wlm1 = jnp.array([jnp.sqrt(3.*np.pi/10), -jnp.sqrt(np.pi/5), jnp.sqrt(3.*np.pi/10)])
    rM1 = G*M1/c**2.
    k2 = 1.5 * lam_M1_5 * (rM1/R1)**5
    Qa1 = jnp.sqrt(k2 * 5/4./np.pi)

    #
    ba1, ba_eq1, va1, Fu1 = \
        get_ba_l2(ff, 
               M1, M2, R1, lam_M1_5, wa01, 
               A_nl_1, w_nl_ref,
               OmegaS1,
               gr_flag)

    # eq. 28 of 2211.07002 for l=2; here we use only the eq tide as it is phase
    # coherent with the orbit
    gr1 = - E1 / (mu * rr) * 3. * 2. * np.real( np.sum(va1 * ba_eq1, axis=0) )
    
    # eq. 39
    dlogr1 = - 1/3. * gr1/g_unit
    # eq. 49
    Eint1 = 2. * dlogr1 * E_orb
    dEorb1 = -2.*Eint1
    
    # eq. 43 of 2211.07002 corrected for spin; see also eq. 58 of 2404.00147
    # this is the only place where we use the full mode amp including dynamical tide 
    # as it is of the form ba * ba.conj instead of ba * va
    Emode1 = E1 * (  (wa1[0] + mm1[0] * OmegaS1)/wa01 * jnp.abs(ba1[0, :])**2. 
                   + (wa1[1] + mm1[1] * OmegaS1)/wa01 * jnp.abs(ba1[1, :])**2.
                   + (wa1[2] + mm1[2] * OmegaS1)/wa01 * jnp.abs(ba1[2, :])**2.)
    
    # 
    dEeq1 = dEorb1 + Eint1 + Emode1

    # eq. 56
    d_dEdt_ns = - 8./15. * eta * M1 * R1**2. / tMt**3. * vv**14.\
        * 2.**6 * jnp.sqrt(3.*np.pi/10) * Qa1 * np.real(ba_eq1[0, :] + ba_eq1[-1, :])
    # eq. 59
    d_dEdt_br = 4. * dlogr1 * dEdt_pp

    d_dEdt_1 = d_dEdt_ns + d_dEdt_br

    ddtdf1 = 1./dEdt_pp * (jnp.gradient(dEeq1)/jnp.gradient(ff) \
                           - jnp.gradient(E_orb)/jnp.gradient(ff) * d_dEdt_1/dEdt_pp)
    
    ddphidf1 = ddtdf1 * 2. * np.pi * ff

    # because we set the initials to 0
    # it won't match exactly the analytical expression in 
    # e.g. Hinderer+ 10, eq. A5
    # this is fine if we remove tc and phic from dpsi in the end
    # dt1 = integ.cumulative_trapezoid(ddtdf1, x=ff, initial=0)
    # dphi1 = integ.cumulative_trapezoid(ddphidf1, x=ff, initial=0)
    dt1 = cumulative_trapezoid(ddtdf1, x=ff, initial=0)
    dphi1 = cumulative_trapezoid(ddphidf1, x=ff, initial=0)

    dpsi1 = 2. * np.pi * ff * dt1 - dphi1

    
    # modes in M2
    ll2 = jnp.array([2, 2, 2])
    mm2 = jnp.array([2, 0, -2])
    # co-rot frame freq
    wa2 = - mm2/ll2 * OmegaS2 + wa02
    
    Wlm2 = jnp.array([jnp.sqrt(3.*np.pi/10), -jnp.sqrt(np.pi/5), jnp.sqrt(3.*np.pi/10)])
    rM2 = G*M2/c**2.
    k2 = 1.5 * lam_M2_5 * (rM2/R2)**5
    Qa2 = jnp.sqrt(k2 * 5/4./np.pi)

    #
    ba2, ba_eq2, va2, Fu2 = \
        get_ba_l2(ff, 
               M2, M1, R2, lam_M2_5, wa02, 
               A_nl_2, w_nl_ref,
               OmegaS2,
               gr_flag)

    # eq. 28 of 2211.07002 for l=2; here we use only the eq tide as it is phase coherent with the orbit
    gr2 = - E2 / (mu * rr) * 3. * 2. * np.real( np.sum(va2 * ba_eq2, axis=0) )
    
    # eq. 39
    dlogr2 = - 1/3. * gr2/g_unit
    # eq. 49
    Eint2 = 2. * dlogr2 * E_orb
    dEorb2 = -2.*Eint2
    
    # eq. 43 of 2211.07002 corrected for spin; see also eq. 58 of 2404.00147
    # this is the only place where we use the full mode amp including dynamical tide 
    # as it is of the form ba * ba.conj instead of ba * va
    Emode2 = E2 * (  (wa2[0] + mm2[0] * OmegaS2)/wa02 * jnp.abs(ba2[0, :])**2. 
                   + (wa2[1] + mm2[1] * OmegaS2)/wa02 * jnp.abs(ba2[1, :])**2.
                   + (wa2[2] + mm2[2] * OmegaS2)/wa02 * jnp.abs(ba2[2, :])**2.)
    
    # 
    dEeq2 = dEorb2 + Eint2 + Emode2

    # eq. 56
    d_dEdt_ns = - 8./15. * eta * M2 * R2**2. / tMt**3. * vv**14.\
        * 2.**6 * jnp.sqrt(3.*np.pi/10) * Qa2 * np.real(ba_eq2[0, :] + ba_eq2[-1, :])
    # eq. 59
    d_dEdt_br = 4. * dlogr2 * dEdt_pp

    d_dEdt_2 = d_dEdt_ns + d_dEdt_br

    ddtdf2 = 1./dEdt_pp * (jnp.gradient(dEeq2)/jnp.gradient(ff) \
                           - jnp.gradient(E_orb)/jnp.gradient(ff) * d_dEdt_2/dEdt_pp)
    ddphidf2 = ddtdf2 * 2. * np.pi * ff

    # dt2 = integ.cumulative_trapezoid(ddtdf2, x=ff, initial=0)
    # dphi2 = integ.cumulative_trapezoid(ddphidf2, x=ff, initial=0)
    dt2 = cumulative_trapezoid(ddtdf2, x=ff, initial=0)
    dphi2 = cumulative_trapezoid(ddphidf2, x=ff, initial=0)

    dpsi2 = 2. * np.pi * ff * dt2 - dphi2

    return dpsi1 + dpsi2

# @jit
def get_f_upper(args_model: dict, roverR_coal=1.05):
    """
    """
    Mchirp, q = args_model['Mchirp'], args_model['q']
    R1, R2 = args_model['R1'], args_model['R2']

    return 1/np.pi * jnp.sqrt(G*Ms*Mchirp*q**(-3/5)*(1+q)**(6/5)/(roverR_coal*(R1+R2))**3)

@jit
def updated_chis(wf_args):
    """
    Update chi1 and chi2 using the I-Love Universal Relation to match S1 and 
    S2

    Returns jnp.array([chi1, chi2])
    """
    S1_and_S2 = jnp.array([wf_args['Omega_S1'], wf_args['Omega_S2']]) 
    # if not any(S1_and_S2):
    #     return jnp.array([0., 0.])
    # else:
    M_cano, lam_cano = wf_args['M_cano'], wf_args['lam_cano']
    Mchirp, q = wf_args['Mchirp'], wf_args['q']
    M1_and_M2 = M1M2fromMchq(Mchirp, q)
    lam1_and_lam2 = Lambda_x(M1_and_M2, M_cano, lam_cano)
    Ibar1_and_Ibar2 = inertia_UR(jnp.log(lam1_and_lam2))
    return S1_and_S2 * (Ibar1_and_Ibar2 * M1_and_M2 *Ms*G/c**3) *2*np.pi

@jit
def updated_Omega_Ss(wf_args):
    """
    Update Omega_S1 and Omega_S2 using the I-Love Universal Relation to match 
    chi1 and chi2

    Returns jnp.array([Omega_S1, Omega_S2])
    """
    # chi1_and_chi2 = jnp.array([wf_args['chi1'], wf_args['chi2']])
    chi1_and_chi2 = get_chi1_chi2(
        wf_args['q'], 
        wf_args['chiP'], 
        wf_args['chiM']
    )
    # if not any(chi1_and_chi2):
    #     return jnp.array([0., 0.])
    # else:
    M_cano, lam_cano = wf_args['M_cano'], wf_args['lam_cano']
    Mchirp, q = wf_args['Mchirp'], wf_args['q']
    M1_and_M2 = M1M2fromMchq(Mchirp, q)
    lam1_and_lam2 = Lambda_x(M1_and_M2, M_cano, lam_cano)
    Ibar1_and_Ibar2 = inertia_UR(jnp.log(lam1_and_lam2))
    return chi1_and_chi2 / (Ibar1_and_Ibar2 * M1_and_M2 *Ms*G/c**3 *2*np.pi)

@jit
def get_chiP_chiM(q, chi1, chi2):
    """
    """
    # q, chi1, chi2 = model_args['q'], model_args['chi1'], model_args['chi2']
    # return jnp.array([q*chi1 + chi2, q*chi1 - chi2])/(1+q)
    # return jnp.array([(q*chi1 + chi2)/(1+q), (chi1 - chi2)])
    return jnp.array([(q*chi1 + chi2)/(1+q), (chi1 - chi2)/2])

@jit
def get_chi1_chi2(q, chiP, chiM):
    """
    """
    # return jnp.array([(chiP+chiM)/q, chiP-chiM])*(1+q)/2
    # return jnp.array([chiP+(chiM/(1+q)), chiP-(chiM*q/(1+q))])*(1+q)/2
    # return jnp.array([chiP+(2*chiM/(1+q)), chiP-(2*chiM*q/(1+q))])*(1+q)
    return jnp.array([(1+q)*chiP + 2*chiM, (1+q)*chiP - 2*chiM*q])/(1+q)

@jit
def rMx(Mx):
    """ Calculate Mass lengthscale
    """
    return G * Mx / c**2

@jit
def tMx(Mx):
    """ Calculate Mass timescale
    """
    return G * Mx / c**3

@jit
def get_Mchirp(M1, M2):
    """
    """
    return (M1**3 * M2**3 / (M1+M2))**(1/5)

@jit
def M1M2fromMchq(Mchirp, q):
    """ Return M1 and M2 in np.array from Mchirp and q (mass ratio)
    """
    M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    return jnp.array([M1, M2])

@jit
def M1fromMchq(Mchirp, q):
    """ Return M1 from Mchirp and q (mass ratio)
    """
    return Mchirp * q**(2/5) * (1+q)**(1/5)

@jit
def M2fromMchq(Mchirp, q):
    """ Return M2 from Mchirp and q (mass ratio)
    """
    return Mchirp * q**(-3/5) * (1+q)**(1/5)


def Lambda_x_lin(Mx, Mc, Lc):
    """ Convert canonical Lambda to NS 1 or 2
    """
    return Lc * (6 - 5*(Mx/Mc))

@jit
def Lambda_x(Mx, Mc, Lc):
    """ Convert canonical Lambda to NS 1 or 2
    """
    return Lc * (Mc/Mx)**5


def Omega_x_lin(Mx, Mc, wc):
    """ Convert canonical omega to NS 1 or 2
    """
    return wc * ((3*Mx)/(2*Mc) - 1/2)

@jit
def Omega_x(Mx, Mc, wc):
    """ Convert canonical omega to NS 1 or 2
    """
    return wc * (Mx/Mc)**1.5


def Anl_x_lin(Mx, Mc, Ac, My):
    """ Convert canonical Anl to NS 1 or 2
    """
    return Ac * (1 + (1/(1 + My/Mc) + 1) * (1 - Mx/Mc) )

@jit
def Anl_x(Mx, Mc, Ac, My):
    """ Convert canonical Anl to NS 1 or 2
    """
    return Ac * (My/Mx)*2*Mc/(Mx+My)


def get_dpsi_qchirp(
    freq, # Hz
    M_c, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    tc, # Milliseconds
    phic, # Radians
    lam_c, # Dimensionless
    wab_c, # Dimensionless
    Anl_c = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    S1 = 0, # Hz
    S2 = 0, # Hz
    gr_flag = 0 # 0 or 1
):
    """ Get dPsi with canonical parameters
    """
    # Calculate M1 and M2
    M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    
    # Convert units
    M_c_g = M_c *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    # tc_sec = tc*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*S1 # Convert to radians    
    Omega_S2_rad = 2*np.pi*S2 # Convert to radians    
    
    lam1 = Lambda_x(M1, M_c, lam_c)
    lam2 = Lambda_x(M2, M_c, lam_c)
    
    wa01 = Omega_x(M1, M_c, wab_c)/tMx(M1_g)
    wa02 = Omega_x(M2, M_c, wab_c)/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_c, Anl_c, M2)
    Anl2 = Anl_x(M2, M_c, Anl_c, M1)
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )
    
    return dpsi


def get_psi(
    freq, # Hz
    M_c, # Solar Mass
    M1, R1, # Solar Mass, Centimeters
    M2, R2, # Solar Mass, Centimeters
    tc, # Milliseconds
    phic, # Radians
    lam_c, # Dimensionless
    wab_c, # Dimensionless
    Anl_c = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    S1 = 0, # Hz
    S2 = 0, # Hz
    gr_flag = 0 # 0 or 1
):
    """ Get Psi with canonical parameters
    """
    # Convert units
    # M_c_g, M1_g, M2_g = np.array([M_c, M1, M2])*Ms # Convert to grams
    M_c_g = M_c *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    tc_sec = tc*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*S1 # Convert to radians    
    Omega_S2_rad = 2*np.pi*S2 # Convert to radians    
    
    lam1 = Lambda_x(M1, M_c, lam_c)
    lam2 = Lambda_x(M2, M_c, lam_c)
    
    wa01 = Omega_x(M1, M_c, wab_c)/tMx(M1_g)
    wa02 = Omega_x(M2, M_c, wab_c)/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_c, Anl_c, M2)
    Anl2 = Anl_x(M2, M_c, Anl_c, M1)
    
    psi_pp = get_psi_pp_newt(
        freq=freq,
        Mchirp=get_Mchirp(M1_g, M2_g),
        q=M1/M2,
        tc=tc_sec,
        phic=phic
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )
    
    psi = psi_pp + dpsi
    return psi


def get_psi_qchirp(
    freq, # Hz
    M_c, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    tc, # Milliseconds
    phic, # Radians
    lam_c, # Dimensionless
    wab_c, # Dimensionless
    Anl_c = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    S1 = 0, # Hz
    S2 = 0, # Hz
    gr_flag = 0 # 0 or 1
):
    """ Get Psi with canonical parameters
    """
    # Calculate M1 and M2
    M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    
    # Convert units
    M_c_g = M_c *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    tc_sec = tc*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*S1 # Convert to radians    
    Omega_S2_rad = 2*np.pi*S2 # Convert to radians    
    
    lam1 = Lambda_x(M1, M_c, lam_c)
    lam2 = Lambda_x(M2, M_c, lam_c)
    
    wa01 = Omega_x(M1, M_c, wab_c)/tMx(M1_g)
    wa02 = Omega_x(M2, M_c, wab_c)/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_c, Anl_c, M2)
    Anl2 = Anl_x(M2, M_c, Anl_c, M1)
    
    psi_pp = get_psi_pp_newt(
        freq=freq,
        Mchirp=Mchirp*Ms,
        q=q,
        tc=tc_sec,
        phic=phic
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )
    
    psi = psi_pp + dpsi
    return psi

#-----------------------------------------------------------------------------

@jit
def get_psi_taylorF2_qchirp(
    freq, # Hz Frequency vector must be linearly spaced
    M_cano, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    t_coal, # Milliseconds
    phi_coal, # Radians
    lam_cano, # Dimensionless
    wab_cano, # Dimensionless
    Anl_cano = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    Omega_S1 = 0, # Hz
    Omega_S2 = 0, # Hz
    chi1 = 0, # Dimensionless z component of spin
    chi2 = 0, # Dimensionless z component of spin
    gr_flag = 0, # 0 or 1
):
    """ Get Psi with canonical parameters
    """
    # Calculate M1 and M2
    M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    
    # Convert units
    M_cano_g = M_cano *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    t_coal_sec = t_coal*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*Omega_S1 # Convert to radians    
    Omega_S2_rad = 2*np.pi*Omega_S2 # Convert to radians    
    
    lam1 = Lambda_x(M1, M_cano, lam_cano)
    lam2 = Lambda_x(M2, M_cano, lam_cano)

    # if UR_lam_wab:
    #     wa01 = omega_UR(jnp.log(lam1))/tMx(M1_g)
    #     wa02 = omega_UR(jnp.log(lam2))/tMx(M2_g)
    # else:
    wa01 = Omega_x(M1, M_cano, wab_cano)/tMx(M1_g)
    wa02 = Omega_x(M2, M_cano, wab_cano)/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_cano, Anl_cano, M2)
    Anl2 = Anl_x(M2, M_cano, Anl_cano, M1)
    
    psi_pp = get_psi_pp_taylorF2_ripple(
        freq=freq, # Hz Frequency vector must be linearly spaced
        # Mchirp=Mchirp, # Solar mass
        # q=q,
        # m1_SI=M1_g*1e-3, # kg
        # m2_SI=M2_g*1e-3, # kg
        M1=M1,
        M2=M2,
        # t_coal=t_coal_sec, # sec
        # phi_coal=phi_coal, # rad
        chi1=chi1,
        chi2=chi2,
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )
    
    psi = psi_pp + dpsi + (2 * np.pi * freq * t_coal_sec) - phi_coal
    return psi


@jit
def get_psi_taylorF2_qchirp_URlw(
    freq, # Hz Frequency vector must be linearly spaced
    M_cano, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    t_coal, # Milliseconds
    phi_coal, # Radians
    lam_cano, # Dimensionless
    wab_cano, # Dimensionless
    Anl_cano = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    Omega_S1 = 0, # Hz
    Omega_S2 = 0, # Hz
    chi1 = 0, # Dimensionless z component of spin
    chi2 = 0, # Dimensionless z component of spin
    gr_flag = 0, # 0 or 1
):
    """ Get Psi with canonical parameters
    """
    # Calculate M1 and M2
    M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    
    # Convert units
    M_cano_g = M_cano *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    t_coal_sec = t_coal*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*Omega_S1 # Convert to radians    
    Omega_S2_rad = 2*np.pi*Omega_S2 # Convert to radians    
    
    lam1 = Lambda_x(M1, M_cano, lam_cano)
    lam2 = Lambda_x(M2, M_cano, lam_cano)

    wa01 = omega_UR(jnp.log(lam1))/tMx(M1_g)
    wa02 = omega_UR(jnp.log(lam2))/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_cano, Anl_cano, M2)
    Anl2 = Anl_x(M2, M_cano, Anl_cano, M1)
    
    psi_pp = get_psi_pp_taylorF2_ripple(
        freq=freq, # Hz Frequency vector must be linearly spaced
        # Mchirp=Mchirp, # Solar mass
        # q=q,
        # m1_SI=M1_g*1e-3, # kg
        # m2_SI=M2_g*1e-3, # kg
        M1=M1,
        M2=M2,
        # t_coal=t_coal_sec, # sec
        # phi_coal=phi_coal, # rad
        chi1=chi1,
        chi2=chi2,
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )
    
    psi = psi_pp + dpsi + (2 * np.pi * freq * t_coal_sec) - phi_coal
    return psi


@jit
def get_h_taylorF2_qchirp(
    freq, # Hz Frequency vector must be linearly spaced
    M_cano, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    t_coal, # Milliseconds
    phi_coal, # Radians
    lam_cano, # Dimensionless
    wab_cano, # Dimensionless
    Anl_cano = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    Omega_S1 = 0, # Hz
    Omega_S2 = 0, # Hz
    chi1 = 0, # Dimensionless z component of spin
    chi2 = 0, # Dimensionless z component of spin
    gr_flag = 0, # 0 or 1
    # UR_lam_wab = False, # Enforce Universal Relation: wab_i(lam_i)
    # UR_lam_I = False # Enforce Universal Relation: Omega_Si(lam_i, chi_i)
):
    """ Get Psi with canonical parameters
    """
    # Calculate M1 and M2
    # M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    # M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    M1, M2 = M1M2fromMchq(Mchirp, q)
    
    # Convert units
    M_cano_g = M_cano *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    t_coal_sec = t_coal*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*Omega_S1 # Convert to radians    
    Omega_S2_rad = 2*np.pi*Omega_S2 # Convert to radians    
    
    lam1 = Lambda_x(M1, M_cano, lam_cano)
    lam2 = Lambda_x(M2, M_cano, lam_cano)

    # if UR_lam_wab:
    #     wa01 = omega_UR(jnp.log(lam1))/tMx(M1_g)
    #     wa02 = omega_UR(jnp.log(lam2))/tMx(M2_g)
    # else:
    wa01 = Omega_x(M1, M_cano, wab_cano)/tMx(M1_g)
    wa02 = Omega_x(M2, M_cano, wab_cano)/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_cano, Anl_cano, M2)
    Anl2 = Anl_x(M2, M_cano, Anl_cano, M1)
    
    h_pp = get_h_pp_taylorF2_ripple(
        freq=freq, # Hz Frequency vector must be linearly spaced
        # Mchirp=Mchirp, # Solar mass
        # q=q,
        # m1_SI=M1_g*1e-3, # kg
        # m2_SI=M2_g*1e-3, # kg
        M1=M1,
        M2=M2,
        # t_coal=t_coal_sec, # sec
        # phi_coal=phi_coal, # rad
        chi1=chi1,
        chi2=chi2,
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )

    exp_dpsi = jnp.exp(1j*(dpsi + 2*np.pi*freq*t_coal_sec - phi_coal))
    
    h_tot = h_pp * exp_dpsi
    return h_tot


@jit
def get_h_taylorF2_qchirp_URlw(
    freq, # Hz Frequency vector must be linearly spaced
    M_cano, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    t_coal, # Milliseconds
    phi_coal, # Radians
    lam_cano, # Dimensionless
    wab_cano, # Dimensionless
    Anl_cano = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    Omega_S1 = 0, # Hz
    Omega_S2 = 0, # Hz
    chi1 = 0, # Dimensionless z component of spin
    chi2 = 0, # Dimensionless z component of spin
    gr_flag = 0, # 0 or 1
):
    """ Get Psi with canonical parameters
    """
    # Calculate M1 and M2
    M1, M2 = M1M2fromMchq(Mchirp, q)
    
    # Convert units
    M_cano_g = M_cano *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    t_coal_sec = t_coal*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*Omega_S1 # Convert to radians    
    Omega_S2_rad = 2*np.pi*Omega_S2 # Convert to radians    
    
    lam1 = Lambda_x(M1, M_cano, lam_cano)
    lam2 = Lambda_x(M2, M_cano, lam_cano)

    wa01 = omega_UR(jnp.log(lam1))/tMx(M1_g)
    wa02 = omega_UR(jnp.log(lam2))/tMx(M2_g)

    Anl1 = Anl_x(M1, M_cano, Anl_cano, M2)
    Anl2 = Anl_x(M2, M_cano, Anl_cano, M1)
    
    h_pp = get_h_pp_taylorF2_ripple(
        freq=freq, # Hz Frequency vector must be linearly spaced
        M1=M1,
        M2=M2,
        chi1=chi1,
        chi2=chi2,
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )

    exp_dpsi = jnp.exp(1j*(dpsi + 2*np.pi*freq*t_coal_sec - phi_coal))
    
    h_tot = h_pp * exp_dpsi
    return h_tot

#-----------------------------------------------------------------------------

@jit
def get_psi_taylorF2_qchirpXeff(
    freq, # Hz Frequency vector must be linearly spaced
    M_cano, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    t_coal, # Milliseconds
    phi_coal, # Radians
    lam_cano, # Dimensionless
    wab_cano, # Dimensionless
    Anl_cano = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    Omega_S1 = 0, # Hz
    Omega_S2 = 0, # Hz
    chiP = 0, # Dimensionless z component of spin
    chiM = 0, # Dimensionless z component of spin
    gr_flag = 0, # 0 or 1
):
    """ Get Psi with canonical parameters
    """
    # Calculate M1 and M2
    M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    
    # Convert units
    M_cano_g = M_cano *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    t_coal_sec = t_coal*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*Omega_S1 # Convert to radians
    Omega_S2_rad = 2*np.pi*Omega_S2 # Convert to radians

    chi1, chi2 = get_chi1_chi2(q, chiP, chiM) # Convert X_eff's to X1, X2
    
    lam1 = Lambda_x(M1, M_cano, lam_cano)
    lam2 = Lambda_x(M2, M_cano, lam_cano)

    wa01 = Omega_x(M1, M_cano, wab_cano)/tMx(M1_g)
    wa02 = Omega_x(M2, M_cano, wab_cano)/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_cano, Anl_cano, M2)
    Anl2 = Anl_x(M2, M_cano, Anl_cano, M1)
    
    psi_pp = get_psi_pp_taylorF2_ripple(
        freq=freq, # Hz Frequency vector must be linearly spaced
        M1=M1,
        M2=M2,
        chi1=chi1,
        chi2=chi2,
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )
    
    psi = psi_pp + dpsi + (2 * np.pi * freq * t_coal_sec) - phi_coal
    return psi


@jit
def get_psi_taylorF2_qchirpXeff_URlw(
    freq, # Hz Frequency vector must be linearly spaced
    M_cano, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    t_coal, # Milliseconds
    phi_coal, # Radians
    lam_cano, # Dimensionless
    wab_cano, # Dimensionless
    Anl_cano = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    Omega_S1 = 0, # Hz
    Omega_S2 = 0, # Hz
    chiP = 0, # Dimensionless z component of spin
    chiM = 0, # Dimensionless z component of spin
    gr_flag = 0, # 0 or 1
):
    """ Get Psi with canonical parameters
    """
    # Calculate M1 and M2
    M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    
    # Convert units
    M_cano_g = M_cano *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    t_coal_sec = t_coal*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*Omega_S1 # Convert to radians
    Omega_S2_rad = 2*np.pi*Omega_S2 # Convert to radians

    chi1, chi2 = get_chi1_chi2(q, chiP, chiM) # Convert X_eff's to X1, X2
    
    lam1 = Lambda_x(M1, M_cano, lam_cano)
    lam2 = Lambda_x(M2, M_cano, lam_cano)

    wa01 = omega_UR(jnp.log(lam1))/tMx(M1_g)
    wa02 = omega_UR(jnp.log(lam2))/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_cano, Anl_cano, M2)
    Anl2 = Anl_x(M2, M_cano, Anl_cano, M1)
    
    psi_pp = get_psi_pp_taylorF2_ripple(
        freq=freq, # Hz Frequency vector must be linearly spaced
        M1=M1,
        M2=M2,
        chi1=chi1,
        chi2=chi2,
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )
    
    psi = psi_pp + dpsi + (2 * np.pi * freq * t_coal_sec) - phi_coal
    return psi


@jit
def get_h_taylorF2_qchirpXeff(
    freq, # Hz Frequency vector must be linearly spaced
    M_cano, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    t_coal, # Milliseconds
    phi_coal, # Radians
    lam_cano, # Dimensionless
    wab_cano, # Dimensionless
    Anl_cano = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    Omega_S1 = 0, # Hz
    Omega_S2 = 0, # Hz
    chiP = 0, # Dimensionless z component of spin
    chiM = 0, # Dimensionless z component of spin
    gr_flag = 0, # 0 or 1
):
    """ Get h with canonical parameters
    """
    # Calculate M1 and M2
    # M1 = Mchirp * q**(2/5) * (1+q)**(1/5)
    # M2 = Mchirp * q**(-3/5) * (1+q)**(1/5)
    M1, M2 = M1M2fromMchq(Mchirp, q)
    
    # Convert units
    M_cano_g = M_cano *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    t_coal_sec = t_coal*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*Omega_S1 # Convert to radians
    Omega_S2_rad = 2*np.pi*Omega_S2 # Convert to radians

    chi1, chi2 = get_chi1_chi2(q, chiP, chiM) # Convert X_eff's to X1, X2
    
    lam1 = Lambda_x(M1, M_cano, lam_cano)
    lam2 = Lambda_x(M2, M_cano, lam_cano)

    # if UR_lam_wab:
    #     wa01 = omega_UR(jnp.log(lam1))/tMx(M1_g)
    #     wa02 = omega_UR(jnp.log(lam2))/tMx(M2_g)
    # else:
    wa01 = Omega_x(M1, M_cano, wab_cano)/tMx(M1_g)
    wa02 = Omega_x(M2, M_cano, wab_cano)/tMx(M2_g)
    
    Anl1 = Anl_x(M1, M_cano, Anl_cano, M2)
    Anl2 = Anl_x(M2, M_cano, Anl_cano, M1)
    
    h_pp = get_h_pp_taylorF2_ripple(
        freq=freq, # Hz Frequency vector must be linearly spaced
        M1=M1,
        M2=M2,
        chi1=chi1,
        chi2=chi2,
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )

    exp_dpsi = jnp.exp(1j*(dpsi + 2*np.pi*freq*t_coal_sec - phi_coal))
    
    h_tot = h_pp * exp_dpsi
    return h_tot


@jit
def get_h_taylorF2_qchirpXeff_URlw(
    freq, # Hz Frequency vector must be linearly spaced
    M_cano, # Solar Mass
    R1, R2, # Centimeters
    Mchirp, # Solar Mass
    q, # Dimensionless
    t_coal, # Milliseconds
    phi_coal, # Radians
    lam_cano, # Dimensionless
    wab_cano, # Dimensionless
    Anl_cano = 0, # Dimensionless
    wnl_ref = np.pi * 1000, # GW Radians
    Omega_S1 = 0, # Hz
    Omega_S2 = 0, # Hz
    chiP = 0, # Dimensionless z component of spin
    chiM = 0, # Dimensionless z component of spin
    gr_flag = 0, # 0 or 1
):
    """ Get Psi with canonical parameters
    """
    # Calculate M1 and M2
    M1, M2 = M1M2fromMchq(Mchirp, q)
    
    # Convert units
    M_cano_g = M_cano *Ms # Convert to grams
    M1_g = M1 *Ms # Convert to grams
    M2_g = M2 *Ms # Convert to grams
    t_coal_sec = t_coal*1e-3 # Convert to seconds
    Omega_S1_rad = 2*np.pi*Omega_S1 # Convert to radians
    Omega_S2_rad = 2*np.pi*Omega_S2 # Convert to radians

    chi1, chi2 = get_chi1_chi2(q, chiP, chiM) # Convert X_eff's to X1, X2
    
    lam1 = Lambda_x(M1, M_cano, lam_cano)
    lam2 = Lambda_x(M2, M_cano, lam_cano)

    wa01 = omega_UR(jnp.log(lam1))/tMx(M1_g)
    wa02 = omega_UR(jnp.log(lam2))/tMx(M2_g)

    Anl1 = Anl_x(M1, M_cano, Anl_cano, M2)
    Anl2 = Anl_x(M2, M_cano, Anl_cano, M1)
    
    h_pp = get_h_pp_taylorF2_ripple(
        freq=freq, # Hz Frequency vector must be linearly spaced
        M1=M1,
        M2=M2,
        chi1=chi1,
        chi2=chi2,
    )
    
    dpsi = get_dpsi(
        freq,
        M1_g, R1, lam1, wa01, 
        M2_g, R2, lam2, wa02, 
        Anl1,
        Anl2,
        wnl_ref,
        Omega_S1_rad,
        Omega_S2_rad,
        gr_flag
    )

    exp_dpsi = jnp.exp(1j*(dpsi + 2*np.pi*freq*t_coal_sec - phi_coal))
    
    h_tot = h_pp * exp_dpsi
    return h_tot

