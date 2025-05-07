import numpy as np



def gain(kappa, g3, g4, alpha_p, omega_p, omega_s, Delta, alpha_s, alpha_i):
    '''
    Frattini et al. Phys. Rev. Appl. (2018), Eq. 3
    '''
    geff = 2*g3*alpha_p

    Deltaeff = Delta + 12*g4* ( (8/9)*np.abs(alpha_p)**2 + np.abs(alpha_s)**2 + np.abs(alpha_i)**2 )

    G = 1 + ( 4*kappa**2*np.abs(geff)**2) / ( Deltaeff )

    return G

def compression1dB()
    '''
    Only considers effect of stark shift
    Frattini et al. Phys. Rev. Appl. (2018), Eq. 4
    '''