import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
import glob, sys

import ccf

def load_parvi_data(filename):
    """
    load filename of parvi spectral file

    0-  f[2].data[0,norder,:] # wavelength Wavelength - andrea inserts the wavelength solution here, right now it is hybrid UNe + LFC wavesol
    1-  Extracted Flux
    2-  Measurement of the error in the extracted flux [1./sqrt(Var)]
    3-  Blaze corrected Flux
    4-  Measurement of the error in the blaze corrected flux
    5-  Corresponding fiber flat extraction
    6-  Error of corresponding fiber flat
    etalon is in extension 4, science in extension 2
    """
    f = fits.open(filename)
    science_extension = 2 # ch3?
    cal_extension = 4 # ch1?

    wave_sci = f[science_extension].data[0, :, :]
    flux_sci = f[science_extension].data[1, :, :]
    err_sci =  f[science_extension].data[2, :, :]

    wave_cal = f[cal_extension].data[0, :, :]
    flux_cal = f[cal_extension].data[1, :, :]
    err_cal =  f[cal_extension].data[2, :, :]

    return [wave_sci, flux_sci, err_sci], [wave_cal, flux_cal, err_cal]

def load_parvi_metadata(filename):
    """load PARVI """
    hdr = fits.getheader(filename)
    return hdr['TIMEWMJD']


def get_palomar_location():
    """EarthLocation for the P200 Hale telescope (matches LAT/LONG/ELEVSEA in PARVI headers)."""
    from astropy.coordinates import EarthLocation
    return EarthLocation.of_site('Palomar')


def compute_airmass_barycorr(mjd, ra_deg, dec_deg, location=None):
    """Compute airmass (sec z) and barycentric RV correction for target coords at given MJD(s).

    mjd:            float or array, MJD (UTC, mid-exposure e.g. header TIMEWMJD)
    ra_deg, dec_deg: target ICRS coordinates in degrees (e.g. from SkyCoord.from_name)
    location:       astropy EarthLocation, defaults to Palomar (P200)

    Returns (airmass, barycorr_kms), each same shape as input mjd.
    barycorr_kms is the velocity to ADD to an observed (topocentric) RV to
    shift it into the solar system barycentric frame.
    """
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, AltAz

    if location is None:
        location = get_palomar_location()

    scalar_input = np.isscalar(mjd)
    time = Time(np.atleast_1d(mjd), format='mjd', scale='utc')
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')

    altaz = target.transform_to(AltAz(obstime=time, location=location))
    airmass = altaz.secz.value

    barycorr = target.radial_velocity_correction(
        kind='barycentric', obstime=time, location=location
    ).to(u.km / u.s).value

    if scalar_input:
        return airmass[0], barycorr[0]
    return airmass, barycorr



def load_wavelength_array(filename='/Altair_R02_20251017031228_deg0_sp.fits'):
    """loads old altair file to get old wavelength solution which shouldn't need to be accurate here
    because parvi data doesn't always have wavelengths stored (TODO to fix this)"""
    f = fits.open(filename)
    science_extension = 2 # ch3?
    cal_extension = 4 # ch1?

    wave_sci = f[science_extension].data[0, :, :]
    wave_cal = f[cal_extension].data[0, :, :]
  
    return wave_sci, wave_cal

def build_master_flux(files, channel):
    """Stack and average flux arrays from a list of files for one channel.

    channel: 'sci' or 'cal'
    Returns master_flux (norders x npix)
    """
    ext = {'sci': 2, 'cal': 4}[channel]
    master = None
    for file in files:
        flux = fits.open(file)[ext].data[1, :, :]
        if master is None:
            master = np.zeros_like(flux)
        master += flux
    return master / len(files)


def make_mask(master_flux, wave, save_to_file=True, file_tag='mask.csv', diagnostics_on=False):
    """Make CCF mask from a single master spectrum.

    master_flux: 2D array (norders x npix)
    wave:        2D array (norders x npix)
    file_tag:    output filename (used if save_to_file=True)
    Returns all_cens, all_weights, mask_file
    """
    from scipy.signal import find_peaks

    norders, npix = np.shape(master_flux)
    mask_file = file_tag if save_to_file else None
    norm_factor = 100000  # so weights are between 0 and 1 for easier reading

    if save_to_file:
        f = open(mask_file, 'w')

    all_cens = {}
    all_weights = {}

    for iorder in np.arange(norders):
        if iorder == 0:  # order 0 doesn't have wave vals
            continue
        ipeaks, _ = find_peaks(master_flux[iorder], prominence=10 * np.nanmedian(master_flux[iorder]), width=3, distance=10)

        if len(ipeaks) > 0:
            if np.nanmedian(np.sqrt(master_flux[iorder][ipeaks])) < 100:
                all_cens[iorder] = []
                all_weights[iorder] = []
            else:
                all_cens[iorder] = wave[iorder][ipeaks]
                weights = np.sqrt(np.maximum(master_flux[iorder][ipeaks], 0) / norm_factor)
                weights[np.isnan(weights)] = 0
                all_weights[iorder] = weights
                if save_to_file:
                    for i in np.arange(len(ipeaks)):
                        f.write(f'{iorder},{all_cens[iorder][i]},{all_weights[iorder][i]}\n')

    if diagnostics_on:
        plt.figure()
        plt.plot(wave[iorder], master_flux[iorder])
        plt.plot(wave[iorder][ipeaks], master_flux[iorder][ipeaks], 'o')
        plt.title(f'Order {iorder}')

    if save_to_file:
        f.close()

    return all_cens, all_weights, mask_file

def load_mask(mask_file):
    """Load a single mask CSV. Returns pandas df with order, cen, weight."""
    import pandas as pd
    return pd.read_csv(mask_file, names=['order', 'cen', 'weight'])


def run_ccf_order(wave, flux, mask, iorder, R=70000, nsamp=3):
    """Run CCF for a single order of a general spectrum.

    wave, flux: 2D arrays (norders x npix)
    mask:       pandas df with order, cen, weight columns
    Returns rv_fit (float, km/s)
    """
    SPEEDOFLIGHT = 299792.458  # km/s
    mask_wid = (SPEEDOFLIGHT / R) / nsamp
    velocity_loop = np.arange(-20, 20, mask_wid / 4)

    mask_order = mask[mask['order'] == iorder]
    if len(mask_order['cen'].values) <= 2:
        return np.nan

    try:
        ccf_out, _ = ccf.ccf_make(
            wave[iorder], flux[iorder],
            mask_order['cen'].values, mask_wid * 10,
            mask_order['weight'].values, velocity_loop,
            R=R, nsamp=nsamp
        )
        rv_fit, _, _, _ = ccf.fit_gaussian_to_ccf_2(
            velocity_loop, 1 - ccf_out / np.max(ccf_out),
            0.1, velocity_halfrange_to_fit=5.0, stddev_guess=1
        )
    except:
        rv_fit = np.nan
        print(f'error in CCF for order {iorder}')

    return rv_fit
