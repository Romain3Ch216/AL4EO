import numpy as np
import csv
from path import get_path

def get_sensor(sensor):
    sensor = Sensor(**SENSORS[sensor])
    return sensor

SENSORS = {
    'ROSIS': {'name': 'Rosis',
             'bands': None,
             'masked_bands': None,
             'wavelengths': None,
             'wv_unit': None,
             'rgb_bands': (55, 41, 12),
             'GSD': None},

    'FENIX': {'name': 'Fenix',
             'bands':'{}/Sensors/Fenix/fenix_selected_bands.npy'.format(get_path()),
             'masked_bands': '{}/Sensors/Fenix/aisa_fenix_masked_bands.npy'.format(get_path()),
             'wavelengths': '{}/Sensors/Fenix/fenix_wavelengths.npy'.format(get_path()),
             'wv_unit': 'µm',
             'rgb_bands': (70, 50, 25),
             'GSD': 55},

    'AVIRIS': {'name': 'Aviris',
             'bands': None,
             'masked_bands': '{}/Sensors/AVIRIS/mask_bands.npy'.format(get_path()),
             'wavelengths': '{}/Sensors/AVIRIS/wavelengths.npy'.format(get_path()),
             'wv_unit': None,
             'rgb_bands': (50, 20, 10),
             'GSD': None},

    'ITRES': {'name': 'ITRES',
             'bands': '{}/Sensors/ITER/bands.npy'.format(get_path()),
             'masked_bands': None,
             'wavelengths': '{}/Sensors/ITER/wavelengths.npy'.format(get_path()),
             'wv_unit': 'nm',
             'rgb_bands': (50, 20, 10),
             'GSD': None},

    'toy_sensor': {'name': 'TOY',
             'bands': None,
             'masked_bands': None,
             'wavelengths': None,
             'wv_unit': None,
             'rgb_bands': (0, 0, 1),
             'GSD': None},
    }


class Sensor:
    def __init__(self, name, bands, masked_bands, wavelengths, wv_unit, rgb_bands, GSD):

        self.name = name
        self.wavelengths = np.load(wavelengths) if wavelengths is not None else None
        self.mask_bands = np.load(masked_bands) if masked_bands is not None else None

        if bands is not None:
            self.bands   = np.load(bands)

        elif self.wavelengths is not None:
            self.bands = np.array(range(1,len(self.wavelengths)+1))

        else:
            self.bands = None

        self.n_bands     = len(self.bands) if self.bands is not None else None
        self.n_tot_bands = len(self.wavelengths) if wavelengths is not None else None
        self.wv_unit     = wv_unit
        self.rgb_bands   = rgb_bands
        self.GSD         = GSD

        if self.n_tot_bands is not None and self.bands is not None:
            mask = np.array([False]*self.n_tot_bands)
            for i in range(len(self.bands)):
                mask[self.bands[i]-1] = True
            self.mask = mask
        else:
            self.mask = None
