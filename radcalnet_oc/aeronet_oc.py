'''
Readers of the AERONET and AERONET-OC files (see :doc:`/processing_chain` for the preparation of the
input of the simulation).

The AERONET-OC data are shared with the other packages through the AERONET-OC database
(``aeronet_oc_db``, optional dependency: ``pip install radcalnet_oc[aeronet]``): location from
``$AERONET_OC_ROOT`` or ``~/.config/aeronet_oc_db/config.toml``, updated with ``aeronet-oc-db update``.
Without it, the site table packaged with radcalnet_oc is used and the files are read from explicit paths.
'''

import os
import pandas as pd
import numpy as np
import xarray as xr
import re
import datetime as dt
from scipy.interpolate import interp1d
from importlib.resources import files

# ------------------------------------
# get path of packaged files
# ------------------------------------
dir, filename = os.path.split(__file__)

aeronet_oc_location_file = files('radcalnet_oc.data.aeronet'
                                 ).joinpath('info_aeronet_oc_sentinel2_location.csv')

# conversion factors of the AERONET-OC columns into the units of the processor input (kg m-2)
TCWV_CM = 10.             # precipitable water, cm -> kg m-2
TCO3_DU = 2.14485e-5      # ozone, Dobson units -> kg m-2
TCNO2_DU = 2.055482e-5    # nitrogen dioxide, Dobson units -> kg m-2


def _import_aeronet_oc_db():
    try:
        import aeronet_oc_db
    except ImportError as e:
        raise ImportError('the AERONET-OC database package aeronet_oc_db is not installed: install it with '
                          '"pip install git+https://github.com/Tristanovsk/aeronet_oc_db.git" (or '
                          '"pip install radcalnet_oc[aeronet]"), set its location ($AERONET_OC_ROOT or '
                          '~/.config/aeronet_oc_db/config.toml) and download the data with '
                          '"aeronet-oc-db update"') from e
    return aeronet_oc_db


def aeronet_inputs(site, level='lev20', start=None, end=None):
    '''
    Input variables of the processor from the AERONET-OC measurements of a site (time series).

    The data are read from the AERONET-OC database (``aeronet_oc_db``, optional dependency) and converted into
    the variables and units of the input of :py:class:`~radcalnet_oc.process.Process` (see
    :doc:`/processing_chain`):

    - ``tcwv``, ``tco3``, ``tcno2`` (kg m-2), ``pressure`` (hPa), ``wind`` (m s-1);
    - ``aot550``, ``aot865`` (AOT interpolated linearly in wavelength) and ``ang_exp`` (550-865 nm);
    - ``naot``: AOT normalized at 550 nm at the AERONET-OC bands (dimension ``wl_aeronet``), for
      :py:class:`~radcalnet_oc.kernel.Aerosol`;
    - ``chl`` (mg m-3), ``sza_aeronet`` (deg) and ``Rrs_Lwn`` (sr-1, dimension ``wl``: the AERONET-OC bands),
      for the hyperspectral reconstruction :py:class:`~radcalnet_oc.reconstruction.Reconstruction`.

    The input of :py:class:`~radcalnet_oc.process.Process` is then completed with the hyperspectral ``Rrs``,
    ``sza`` and ``day_of_year`` (see the tutorials).

    :param site: AERONET-OC site name of the database (e.g. ``'AAOT'``, ``'Section-7_Platform'``)
    :param level: ``'lev15'`` or ``'lev20'``
    :param start: first time (anything accepted by :py:meth:`xarray.Dataset.sel`, e.g. ``'2024-07-02'``),
        or None for the beginning of the record
    :param end: last time (a date includes the whole day), or None for the end of the record
    :return: :py:class:`xarray.Dataset` (dimension ``time``, and ``wl``, ``wl_aeronet`` for the spectral
        variables)
    :raises ImportError: if ``aeronet_oc_db`` is not installed
    '''
    aeronet_oc_db = _import_aeronet_oc_db()
    # the selection is read in memory and the file closed (no file handle left open)
    with aeronet_oc_db.open_site(site, level, legacy_names=True) as ds:
        if start is not None or end is not None:
            ds = ds.sel(time=slice(start, end))
        ds = ds.load().sortby('wl')
    aod = ds['Aerosol_Optical_Depth'].interpolate_na('wl')
    out = xr.Dataset(coords={'time': ds.time})
    out['tcwv'] = ds['Total_Precipitable_Water(cm)'] * TCWV_CM     # kg m-2
    out['tco3'] = ds['Total_Ozone(Du)'] * TCO3_DU                  # kg m-2
    out['tcno2'] = ds['Total_NO2(DU)'] * TCNO2_DU                  # kg m-2
    out['pressure'] = ds['Pressure(hPa)']
    out['wind'] = ds['Wind_Speed(ms)']
    out['aot550'] = aod.interp(wl=550)
    out['aot865'] = aod.interp(wl=865)
    out['ang_exp'] = -np.log(out.aot550 / out.aot865) / np.log(550 / 865)
    out['naot'] = (ds['Aerosol_Optical_Depth'] / out.aot550).rename({'wl': 'wl_aeronet'})
    out['chl'] = ds['Chlorophyll-a']
    out['sza_aeronet'] = ds['SZA']
    out['Rrs_Lwn'] = ds['Rrs_Lwn']
    # attributes of the converted variables (the arithmetic keeps those of the AERONET-OC fields)
    for name, units, long_name in [('tcwv', 'kg m-2', 'total column water vapour'),
                                   ('tco3', 'kg m-2', 'total column ozone'),
                                   ('tcno2', 'kg m-2', 'total column NO2'),
                                   ('aot550', '1', 'aerosol optical thickness at 550 nm'),
                                   ('aot865', '1', 'aerosol optical thickness at 865 nm'),
                                   ('ang_exp', '1', 'Angstrom exponent 550-865 nm'),
                                   ('naot', '1', 'aerosol optical thickness normalized at 550 nm')]:
        source = {k: v for k, v in out[name].attrs.items() if k == 'aeronet_name'}
        out[name].attrs = {'units': units, 'long_name': long_name, **source}
    return out.reset_coords(drop=True)


class Aeronet:
    '''
    Readers of AERONET-OC and AERONET files into :py:class:`pandas.DataFrame`.

    The data are indexed by time; the columns are a 3-level :py:class:`pandas.MultiIndex`: original
    field name, data type (e.g. ``'aot'``, ``'wavelength'``) and wavelength (nm) for the spectral fields.

    :ivar aeronet_oc_site_info: :py:class:`pandas.DataFrame` of the AERONET-OC sites and their
        location in the Sentinel-2 tiles
    '''

    def __init__(self):
        '''
        Load the information of the AERONET-OC sites.
        '''
        self.get_aeronet_oc_site_info()

    def get_aeronet_oc_site_info(self):
        '''
        Load the AERONET-OC sites and their location in the Sentinel-2 tiles into ``aeronet_oc_site_info``
        (index ``site``, columns ``latitude``, ``longitude``, ``s2_tile``): from the AERONET-OC database if
        it is installed and built, else from the table packaged with radcalnet_oc.
        '''
        info = None
        try:
            import aeronet_oc_db
            info = aeronet_oc_db.site_info()
        except (ImportError, OSError):
            pass
        if info is None or info.empty:
            info = pd.read_csv(aeronet_oc_location_file, index_col=0)
        self.aeronet_oc_site_info = info[['latitude', 'longitude', 's2_tile']]

    def read_site(self, site, level='lev20'):
        '''
        Read the AERONET-OC file of a site from the AERONET-OC database (raw file, kept up to date by
        ``aeronet-oc-db update``), see :meth:`read_aeronet_ocv3`.

        :param site: AERONET site name (e.g. ``'AAOT'``)
        :param level: ``'lev15'`` or ``'lev20'``
        '''
        import aeronet_oc_db
        return self.read_aeronet_ocv3(str(aeronet_oc_db.raw_file(site, level)))

    @staticmethod
    def open_site(site, level='lev20', legacy_names=False):
        '''
        :py:class:`xarray.Dataset` (time, wl) of a site from the AERONET-OC database, including the remote
        sensing reflectances ``Rrs_Lwn``, ``Rrs_Lwn_fQ`` and ``Rrs_Lwn_IOP`` (see ``aeronet_oc_db``).

        :param legacy_names: variable names of the former ``AERONET/netcdf/OCv3`` files
        '''
        import aeronet_oc_db
        return aeronet_oc_db.open_site(site, level, legacy_names=legacy_names)



    def read_aeronet_ocv3(self,
                          ifile,
                          skiprows=8,
                          encoding='latin_1'):
        '''
        Read an AERONET-OC version 3 file (e.g. ``AAOT_OCv3.lev20``).

        :param ifile: path of the AERONET-OC file
        :param skiprows: number of header rows to skip
        :param encoding: encoding of the file
        :return: :py:class:`pandas.DataFrame` indexed by time, columns sorted by increasing wavelength
        '''

        h1 = pd.read_csv(ifile, skiprows=skiprows - 1, nrows=1, encoding=encoding).columns[3:]
        h1 = np.insert(h1, 0, 'site')
        data_type = h1.str.replace(r'\[.*\]', '', regex=True)
        data_type = data_type.str.replace(r'Exact_Wave.*', 'wavelength', regex=True)

        # convert into float to order the dataframe with increasing wavelength
        h2 = h1.str.replace(r'.*\[', '', regex=True)
        h2 = h2.str.replace(r'nm\].*', '', regex=True)
        h2 = h2.str.replace(r'Exact_Wavelengths\(um\)_', '', regex=True)
        h2 = pd.to_numeric(h2, errors='coerce')  # h2.str.extract('(\d+)').astype('float')
        h2 = h2.fillna('').T

        df = pd.read_csv(ifile, skiprows=skiprows, header=None,
                         na_values=['N/A', -999.0, -9.999999], index_col=False, encoding=encoding)

        df['time'] = pd.to_datetime(df.pop(1).astype(str) + df.pop(2).astype(str), format="%d:%m:%Y%H:%M:%S")
        # df['site'] = site
        # df.set_index(['site', 'time'],inplace=True)
        df.set_index('time', inplace=True)

        tuples = list(zip(h1, data_type, h2))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.columns = pd.MultiIndex.from_tuples([(x[0], x[1], x[2]) for x in df.columns])
        df.sort_index(axis=1, level=2, inplace=True)
        return df

    def read_aeronet(self,
                     skiprows=6,
                     encoding='latin_1'):
        '''
        Read an AERONET version 3 aerosol optical depth file, whose path is given by the ``file``
        attribute.

        :param skiprows: number of header rows to skip
        :param encoding: encoding of the file
        :return: :py:class:`pandas.DataFrame` indexed by time; the exact wavelengths are converted to nm
        '''

        ifile = self.file
        df = pd.read_csv(ifile, skiprows=skiprows, nrows=1, encoding=encoding)  # read just first line for columns
        columns = df.columns.tolist()  # get the columns
        cols_to_use = columns[:len(columns) - 1]  # drop the last one
        df = pd.read_csv(ifile,
                         skiprows=skiprows,
                         usecols=cols_to_use,
                         index_col=False,
                         na_values=['N/A', -999.0],
                         encoding=encoding)

        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.rename(columns={'AERONET_Site_Name': 'site', 'Last_Processing_Date(dd/mm/yyyy)': 'Last_Processing_Date'},
                  inplace=True)
        format = "%d:%m:%Y%H:%M:%S"
        df['time'] = pd.to_datetime(df[df.columns[0]] + df[df.columns[1]], format=format)
        # df.set_index(['site','time'], inplace=True)
        df.set_index('time', inplace=True)
        df = df.drop(df.columns[[0, 1]], axis=1)
        # df['year'] = df.index.get_level_values(1).year

        # cleaning up
        df.drop(list(df.filter(regex='Input')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Empty')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Day')), axis=1, inplace=True)

        # indexing columns with spectral values
        data_type = df.columns.str.replace(r'AOD.*nm', 'aot', regex=True)
        data_type = data_type.str.replace(r'Exact_Wave.*', 'wavelength', regex=True)
        data_type = data_type.str.replace(r'Triplet.*[0-9]', 'std', regex=True)
        data_type = data_type.str.replace(r'^(?!aot|std|wavelength).*$', '', regex=True)

        wl_type = df.columns.str.extract(r'(\d+)').astype('float')
        wl_type = wl_type.fillna('')

        tuples = list(zip(df.columns, data_type, wl_type))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        if 'wavelength' in df.columns.levels[1]:
            df.loc[:, (slice(None), 'wavelength',)] = df.loc[:, (slice(None), 'wavelength')] * 1000  # convert into nm
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.sort_index(axis=1, level=2, inplace=True)
        return df

    def read_aeronet_inv(self,
                         skiprows=6,
                         encoding='latin_1'):
        '''
        Read an AERONET version 3 aerosol inversion file, whose path is given by the ``file`` attribute.

        :param skiprows: number of header rows to skip
        :param encoding: encoding of the file
        :return: :py:class:`pandas.DataFrame` indexed by time; the exact wavelengths are converted to nm
        '''
        ifile = self.file
        df = pd.read_csv(ifile, skiprows=skiprows, nrows=1,encoding=encoding)  # read just first line for columns
        columns = df.columns.tolist()  # get the columns
        cols_to_use = columns[:len(columns) - 1]  # drop the last one
        df = pd.read_csv(ifile,
                         skiprows=skiprows,
                         usecols=cols_to_use,
                         index_col=False,
                         na_values=['N/A', -999.0],
                         encoding=encoding)

        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.rename(columns={'AERONET_Site_Name': 'site', 'Last_Processing_Date(dd/mm/yyyy)': 'Last_Processing_Date', },
                  inplace=True)
        format = "%d:%m:%Y %H:%M:%S"
        df['time'] = pd.to_datetime(df[df.columns[1]] + ' ' + df[df.columns[2]], format=format)
        # df.set_index(['site','time'], inplace=True)
        df.set_index('time', inplace=True)
        df = df.drop(df.columns[[0, 1]], axis=1)
        # df['year'] = df.index.get_level_values(1).year

        # cleaning up
        df.drop(list(df.filter(regex='Input')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Empty')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Day')), axis=1, inplace=True)
        df.drop(list(df.filter(regex='Angle_Bin')), axis=1, inplace=True)

        # indexing columns with spectral values
        data_type = df.columns.str.replace(r'AOD.*nm', 'aot')
        data_type = data_type.str.replace(r'Exact_Wave.*', 'wavelength')
        data_type = data_type.str.replace(r'Triplet.*[0-9]', 'std')
        data_type = data_type.str.replace(r'^(?!aot|std|wavelength).*$', '')

        wl_type = df.columns.str.extract(r'(\d+)').astype('float')
        wl_type = wl_type.fillna('')

        tuples = list(zip(df.columns, data_type, wl_type))
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['l0', 'l1', 'l2'])
        if 'wavelength' in df.columns.levels[1]:
            df.loc[:, (slice(None), 'wavelength',)] = df.loc[:, (slice(None), 'wavelength')] * 1000  # convert into nm

        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df.sort_index(axis=1, level=2, inplace=True)

        return df
