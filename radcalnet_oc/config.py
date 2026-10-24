'''
Configuration of radcalnet_oc: settings, folder of the TOA look-up table (LUT) and its download from
Zenodo.

The settings are read from, in increasing order of priority:

1. the default configuration shipped with the package (``radcalnet_oc/config.yml``),
2. the user configuration file: the file given by the ``RADCALNET_OC_CONFIG`` environment variable or,
   if not set, ``$XDG_CONFIG_HOME/radcalnet_oc/config.yml`` (``~/.config/radcalnet_oc/config.yml`` by
   default),
3. the ``RADCALNET_OC_DATA_ROOT`` environment variable, which overrides ``path: lutdata``.

If ``lutdata`` is not set (``null``), the TOA LUT is read from the default folder of the user data,
:py:func:`~radcalnet_oc.config.default_data_root` (``~/.local/share/radcalnet_oc/lut`` on Linux). The TOA LUT is the one of
GRS (same Zenodo record): a GRS LUT folder can be used as ``lutdata``.

The user file only needs the keys that differ from the defaults, e.g.::

    path:
      lutdata: '/path/to/lut'

Create it with ``radcalnet_oc_config --init /path/to/lut``, download the TOA LUT with
``radcalnet_oc_config --download`` and run ``radcalnet_oc_config`` to print the configuration in use
and check the LUT.

The missing TOA LUT is also downloaded automatically when the LUTs are loaded (:py:func:`~radcalnet_oc.config.ensure_toa_lut`),
unless ``zenodo: auto_download: false`` or ``RADCALNET_OC_NO_DOWNLOAD=1``; without it, the light LUT of
the package is used, with a lower accuracy.
'''

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
import urllib.request
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import platformdirs
import yaml

PACKAGE = 'radcalnet_oc'

# light TOA LUT shipped with the package, used when the TOA LUT of the configuration is missing
LUT_FILE_BACKUP = files('radcalnet_oc.data.lut.atmo').joinpath('toa_lut_opac_ultra_light.nc')


def default_data_root():
    '''
    Default folder of the TOA LUT, used when ``lutdata`` is not set: ``lut`` in the user data folder of
    radcalnet_oc (``~/.local/share/radcalnet_oc/lut`` on Linux, ``~/Library/Application
    Support/radcalnet_oc/lut`` on macOS, ``%LOCALAPPDATA%\\radcalnet_oc\\lut`` on Windows).
    '''
    return platformdirs.user_data_path(PACKAGE) / 'lut'


def default_config_file():
    '''Return the packaged default configuration file.'''
    return files(PACKAGE).joinpath('config.yml')


def user_config_file():
    '''Return the path of the user configuration file (which may not exist).'''
    if os.environ.get('RADCALNET_OC_CONFIG'):
        return Path(os.environ['RADCALNET_OC_CONFIG']).expanduser()
    config_home = os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config'
    return Path(config_home) / PACKAGE / 'config.yml'


def read_user_config():
    '''
    Return the settings of the user configuration file, ``{}`` if there is none.

    :raises FileNotFoundError: if ``RADCALNET_OC_CONFIG`` points to a missing file
    '''
    user_file = user_config_file()
    if user_file.is_file():
        with open(user_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file) or {}
    if os.environ.get('RADCALNET_OC_CONFIG'):
        raise FileNotFoundError(f'RADCALNET_OC_CONFIG points to a missing file: {user_file}')
    return {}


def _merge(base, update):
    '''Recursively merge the dictionary `update` into a copy of `base`.'''
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=None)
def load_config():
    '''
    Load the radcalnet_oc configuration (defaults, then user file, then environment variables).

    The result is cached: call ``load_config.cache_clear()`` after changing the user file or the
    environment variables in a running session.
    '''
    with default_config_file().open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}

    config = _merge(config, read_user_config())

    if os.environ.get('RADCALNET_OC_DATA_ROOT'):
        config = _merge(config, {'path': {'lutdata': os.environ['RADCALNET_OC_DATA_ROOT']}})

    lutdata = config['path'].get('lutdata')
    config['path']['lutdata'] = os.path.expanduser(lutdata) if lutdata else str(default_data_root())
    return config


def data_root_source():
    '''Return where the LUT folder in use is set: ``RADCALNET_OC_DATA_ROOT``, user file or default folder.'''
    if os.environ.get('RADCALNET_OC_DATA_ROOT'):
        return 'RADCALNET_OC_DATA_ROOT'
    if (read_user_config().get('path') or {}).get('lutdata'):
        return 'user configuration'
    return 'default folder'


def toa_lut_path():
    '''Path of the TOA LUT of the configuration in use (``lutdata/toa_lut``).'''
    paths = load_config()['path']
    return Path(paths['lutdata'], paths['toa_lut'])


def trans_lut_path():
    '''Path of the irradiance transmittance LUT (``trans_lut``, shipped with the package).'''
    return files('radcalnet_oc.data.lut.atmo').joinpath(load_config()['path']['trans_lut'])


def auto_download_enabled():
    '''
    Return whether the missing TOA LUT is downloaded automatically: ``zenodo: auto_download`` of the
    configuration (default True), switched off by the environment variable ``RADCALNET_OC_NO_DOWNLOAD``
    (any value but ``''``/``0``), e.g. on a cluster without internet access.
    '''
    if os.environ.get('RADCALNET_OC_NO_DOWNLOAD', '') not in ('', '0'):
        return False
    return bool((load_config().get('zenodo') or {}).get('auto_download', True))


def ensure_toa_lut():
    '''
    Return the TOA LUT of the configuration in use, downloading it from Zenodo if it is missing and the
    automatic download is enabled (:py:func:`~radcalnet_oc.config.auto_download_enabled`).

    Called when the LUTs are loaded, so that a fresh installation (``pip install radcalnet_oc``) only
    needs internet access on the first run. The LUT goes to the LUT folder in use (default folder:
    :py:func:`~radcalnet_oc.config.default_data_root`).

    :return: path of the TOA LUT, or None if it is missing and cannot be downloaded
    '''
    path = toa_lut_path()
    if path.is_file():
        return path
    if auto_download_enabled():
        logging.info('TOA LUT missing: downloading it from Zenodo into %s (once, about 900 MB)',
                     load_config()['path']['lutdata'])
        try:
            download_toa_lut()
        except (ValueError, OSError) as error:
            logging.error('automatic download of the TOA LUT failed: %s', error)
    return path if path.is_file() else None


def init_user_config(data_root=None, force=False):
    '''
    Write the user configuration file with the LUT folder `data_root` and, commented, the default
    settings (to uncomment and change if needed).

    :param data_root: folder of the TOA LUT; if None, ``lutdata`` is not set and the default folder
        :py:func:`~radcalnet_oc.config.default_data_root` is used (and created)
    :param force: overwrite an existing user configuration file
    :return: path of the file written
    :raises FileExistsError: if the file exists and `force` is False
    '''
    user_file = user_config_file()
    if user_file.exists() and not force:
        raise FileExistsError(f'{user_file} already exists (use --force to overwrite it)')

    defaults = default_config_file().read_text(encoding='utf-8')
    defaults = defaults.split('\n\n', 1)[1]  # without the header of the packaged file
    commented = ''.join(('# ' + line if line.strip() else '#\n') if not line.startswith('#') else line
                        for line in defaults.splitlines(keepends=True))
    if data_root:
        data_root_lines = f"path:\n  lutdata: '{Path(data_root).expanduser().absolute()}'\n"
    else:
        data_root_lines = (f'# LUT folder: the default one, {default_data_root()}; to use another folder:\n'
                           '# path:\n'
                           "#   lutdata: '/path/to/lut'\n")
    content = (
        f'# radcalnet_oc user configuration, created by `radcalnet_oc_config --init` on {datetime.date.today()}.\n'
        '# Only the keys that differ from the packaged defaults are needed: to change a setting, copy\n'
        '# it (with its parent keys) from the defaults below and uncomment it.\n'
        '# Print the configuration in use and check the LUT with `radcalnet_oc_config`.\n'
        '\n'
        f'{data_root_lines}'
        '\n'
        '# ------------------------------------------------------------------------------------------\n'
        '# Packaged defaults (radcalnet_oc/config.yml)\n'
        '# ------------------------------------------------------------------------------------------\n'
        f'{commented}'
    )
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text(content, encoding='utf-8')
    if not data_root:
        default_data_root().mkdir(parents=True, exist_ok=True)
    load_config.cache_clear()
    return user_file


def zenodo_record_url():
    '''
    URL of the Zenodo API record of the TOA LUT (``zenodo`` section of the configuration).

    :raises ValueError: if no record is configured
    '''
    zenodo = load_config().get('zenodo') or {}
    if not zenodo.get('record'):
        raise ValueError('no Zenodo record of the TOA LUT is configured (zenodo: record); download the '
                         'LUT manually and copy it into the LUT folder (see `radcalnet_oc_config`)')
    return f"{zenodo.get('url', 'https://zenodo.org').rstrip('/')}/api/records/{zenodo['record']}"


def download_toa_lut(force=False, progress=True):
    '''
    Download the TOA LUT of the configuration in use from Zenodo into the LUT folder.

    The file is checked against the checksum given by Zenodo; it is written to a ``.part`` file renamed
    at the end, so that an interrupted download leaves no incomplete LUT.

    :param force: download the file if it already exists
    :param progress: print the progress of the download
    :return: path of the TOA LUT
    '''
    path = toa_lut_path()
    if path.is_file() and not force:
        if progress:
            print(f'TOA LUT already there: {path}')
        return path

    with urllib.request.urlopen(zenodo_record_url()) as response:
        record = json.load(response)
    remote = {f['key']: f for f in record['files']}
    if path.name not in remote:
        raise FileNotFoundError(f'{path.name} is not in the Zenodo record {record["links"].get("self_html", "")}')

    entry = remote[path.name]
    algorithm, expected = entry['checksum'].split(':', 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    digest = hashlib.new(algorithm)
    done = 0
    with urllib.request.urlopen(entry['links']['self']) as response, open(part, 'wb') as out:
        for block in iter(lambda: response.read(1 << 20), b''):
            out.write(block)
            digest.update(block)
            done += len(block)
            if progress:
                print(f'\rTOA LUT downloading {path.name}: {done / 1e6:.0f}/{entry["size"] / 1e6:.0f} MB',
                      end='', flush=True)
    if progress:
        print(flush=True)
    if digest.hexdigest() != expected:
        part.unlink()
        raise IOError(f'{algorithm} checksum mismatch for {path.name}: download it again')
    part.replace(path)
    return path


def main(argv=None):
    '''
    ``radcalnet_oc_config`` command: print the configuration in use and check the TOA LUT (exit code 1
    if it is missing), create the user configuration file with ``--init``, download the TOA LUT with
    ``--download``.
    '''
    parser = argparse.ArgumentParser(
        prog='radcalnet_oc_config',
        description='Print the radcalnet_oc configuration in use and check the TOA LUT, create the user '
                    'configuration file, or download the TOA LUT from Zenodo.')
    parser.add_argument('--init', nargs='?', const='', metavar='DATA_ROOT',
                        help='create the user configuration file, with DATA_ROOT the folder of the TOA LUT '
                             '(default folder if not given)')
    parser.add_argument('--download', action='store_true',
                        help='download the missing TOA LUT from Zenodo into the LUT folder')
    parser.add_argument('--force', action='store_true',
                        help='with --init, overwrite an existing user configuration file; '
                             'with --download, download the TOA LUT again')
    args = parser.parse_args(argv)

    if args.init is not None:
        try:
            user_file = init_user_config(args.init or None, force=args.force)
        except FileExistsError as error:
            print(f'error: {error}', file=sys.stderr)
            return 1
        print(f'user configuration written: {user_file}')
        print('---')

    if args.download:
        try:
            download_toa_lut(force=args.force)
        except (ValueError, OSError) as error:
            print(f'error: {error}', file=sys.stderr)
            return 1
        print('---')

    user_file = user_config_file()
    print(f'default configuration: {default_config_file()}')
    print(f'user configuration:    {user_file} ({"found" if user_file.is_file() else "not found"})')
    print(f'LUT folder:            {load_config()["path"]["lutdata"]} ({data_root_source()})')
    print(f'automatic download:    {"on" if auto_download_enabled() else "off"} '
          '(missing TOA LUT downloaded when the LUTs are loaded)')
    print('---')
    print(yaml.safe_dump(load_config(), sort_keys=False), end='')
    print('---')
    path = toa_lut_path()
    print(f'TOA LUT          {"ok     " if path.is_file() else "MISSING"} {path}')
    print(f'transmittance    ok      {trans_lut_path()} (package)')
    if not path.is_file():
        print(f'error: TOA LUT missing (the light LUT of the package is used, with a lower accuracy); '
              f'download it with `radcalnet_oc_config --download`, or copy it into '
              f'{load_config()["path"]["lutdata"]}, or set its folder with `radcalnet_oc_config --init '
              '/path/to/lut` (or path: lutdata in the user configuration file, or RADCALNET_OC_DATA_ROOT)',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
