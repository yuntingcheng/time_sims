from time_paths import *
from scipy.io.idl import readsav
from cosmo_tools import *

class TIME_param:
    def __init__(self,survey_shape = 'slab'):
        '''
        survey shape:
        'slab': 1D line scan for 180 spatial pixels with 0.45 arcmin pixel size
        
        'cube': 2D square scan for 180x180 pixels
        '''
        assert survey_shape in ['slab','cube'], "survey_shape should be 'slab' or 'cube'"
        
        self.survey_shape = survey_shape
        self.dth = 0.43 # arcmin
        self.Nx = 180
        
        if survey_shape == 'slab': self.Ny = 1
        else: self.Ny = 180
        
        self._get_bins()
        
        # total integration time
        self.t_int = 1000 # Hr
        # 16 spectrometers
        self.N_spec = 16

        self.t_pix = self._t_pix() # [sec]
    
    def _get_bins(self):
        TP_arizona = readsav(mypaths['datadir'] + "TP_arizona.sav")
        nu_bins = np.asarray(TP_arizona['detfreq'])
        self.nu_bins = nu_bins
        
        nu_bins1 = np.append(2*nu_bins[0]-nu_bins[1], nu_bins)
        nu_bins1 = np.append(nu_bins1, 2*nu_bins1[-1]-nu_bins1[-2])
        nu_binedges = (nu_bins1[:-1] + nu_bins1[1:]) / 2
        self.nu_binedges = nu_binedges
        
        self.dnus = nu_binedges[:-1] - nu_binedges[1:]
        self.dnus_data = TP_arizona['detdnu']
        
        self.Nnu = len(nu_bins)
        self.nu_min = min(nu_binedges)
        self.nu_max = max(nu_binedges)
        
        self.science_band = np.asanyarray(TP_arizona['scienceband_high'] + TP_arizona['scienceband_low'])
        nu_bins = nu_bins[self.science_band==1]
        self.nu_bins_science = nu_bins
        nu_bins1 = np.append(2*nu_bins[0]-nu_bins[1], nu_bins)
        nu_bins1 = np.append(nu_bins1, 2*nu_bins1[-1]-nu_bins1[-2])
        nu_binedges = (nu_bins1[:-1] + nu_bins1[1:]) / 2
        self.nu_binedges_science = nu_binedges
        
        return
    
    def _t_pix(self):
        '''
        # pixel integration time
        '''
        Omega_pix = self.dth**2 #[arcmin^2]
        Omega_survey = self.dth**2 * self.Nx * 1. # [arcmin^2]
        t_tot = self.t_int * 3600 # total integration time [sec]
        t_pix = (Omega_pix/Omega_survey) * self.N_spec * t_tot * 2
        return t_pix

    def cmv_config(self, linename, jco = 1):
        _get_cmv_config(self, linename, jco = jco)
        return

def _get_cmv_config(survey, linename, jco = 1):

    if linename =='cii':
        nu_rest = spec_lines.CII.to(u.GHz, equivalencies=u.spectral()).value
        name = 'CII'
    elif linename =='co':
        if jco not in np.arange(1,9,1,dtype=int):
            print('jco data not exist! (jco best be in [1,2,...,8])')
            return
        else:
            nu_rest = spec_lines.CO(jco).to(u.GHz, equivalencies=u.spectral()).value
            name = 'CO(' + str(jco) + '-' + str(jco-1) + ')'
    elif linename == 'Lya':
        nu_rest = spec_lines.Lya.to(u.GHz, equivalencies=u.spectral()).value
        name = 'Lya'
    elif linename == 'Ha':
        nu_rest = spec_lines.Ha.to(u.GHz, equivalencies=u.spectral()).value
        name = 'Ha'
    elif linename == 'Hb':
        nu_rest = spec_lines.Hb.to(u.GHz, equivalencies=u.spectral()).value
        name = 'Hb'
    elif linename == 'OII':
        nu_rest = spec_lines.OII.to(u.GHz, equivalencies=u.spectral()).value
        name = 'OII'
    elif linename == 'OIII':
        nu_rest = spec_lines.OIII.to(u.GHz, equivalencies=u.spectral()).value
        name = 'OIII'
    else:
        print('line name has to be "cii", "co", "Lya", "Ha", "Hb", "OII", "OIII".')
        return

    survey.z_bins = (nu_rest/survey.nu_bins) - 1
    survey.z_binedges = (nu_rest/survey.nu_binedges) - 1

    xDcmv_vec = np.zeros_like(survey.z_bins)
    for i in range(len(xDcmv_vec)):
        xDcmv_vec[i] = cosmo_dist(survey.z_bins[i]).comoving_distance.value
    xDcmv_vec *= (survey.dth * u.arcmin).to(u.rad).value
    survey.xDcmv_vec = xDcmv_vec

    zDcmv_edges_vec = np.zeros_like(survey.z_binedges)
    for i in range(len(zDcmv_edges_vec)):
        zDcmv_edges_vec[i] = cosmo_dist(survey.z_binedges[i]).comoving_distance.value
    survey.zDcmv_vec = zDcmv_edges_vec[1:] - zDcmv_edges_vec[:-1]

    survey.k_p_min = 2 * np.pi / np.sum(survey.xDcmv_vec)
    survey.k_p_max = np.pi / np.mean(survey.xDcmv_vec)
    survey.k_l_min = 2 * np.pi / np.sum(survey.zDcmv_vec)
    survey.k_l_max = np.pi / np.mean(survey.zDcmv_vec)

    return
    
def _get_freq_binedges(nu_min, nu_max, R = None, dnu = None, Nbins = None):
    '''
    Calculate the frequency binedges for given freq min / max and spec resolution, or Nbins.
    
    Inputs:
    =======
    nu_min: min freq [GHz]
    nu_max: max freq [GHz]
    R: spectral resolution nu / dnu
    dnu: spectral resolution dnu [GHz]
    Nbins : # of equally spacing freq bins (bin edges has Nbins+1 dim)
    Outputs:
    ========
    nu_bins: freq bin edges (from low to high freq) [GHz]
    '''
    
    if R != None:
        nu_mid = (nu_min + nu_max) / 2.
        dnu = nu_mid / R
    
    if dnu != None:
        nu_bins = np.arange(nu_min, nu_max + 0.1 * dnu, dnu)
        nu_bins = nu_bins[nu_bins <= nu_max]
    
    if Nbins != None:
        nu_bins = np.linspace(nu_min, nu_max, Nbins+1)
    
    return nu_bins