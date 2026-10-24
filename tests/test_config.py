"""Configuration: packaged defaults, user file, environment variables and download of the TOA LUT."""

import hashlib
import io
import json
import os
import urllib.error
import urllib.request

import pytest
import yaml

from radcalnet_oc import config

LUT_NAME = 'toa_lut_opac_wind_v3.nc'
FILE_URL = 'https://zenodo.test/files/' + LUT_NAME


def fake_zenodo(monkeypatch, content, checksum=None, key=LUT_NAME):
    """Replace the Zenodo API by a record holding one file `key` of content `content`; return the list of
    the URLs requested."""
    record = {'files': [{'key': key,
                         'size': len(content),
                         'checksum': checksum or 'md5:' + hashlib.md5(content).hexdigest(),
                         'links': {'self': FILE_URL}}],
              'links': {'self_html': 'https://zenodo.test/records/23394202'}}
    calls = []

    def urlopen(url):
        calls.append(url)
        if url == 'https://zenodo.org/api/records/23394202':
            return io.BytesIO(json.dumps(record).encode())
        if url == FILE_URL:
            return io.BytesIO(content)
        raise AssertionError(f'unexpected URL {url}')

    monkeypatch.setattr(config.urllib.request, 'urlopen', urlopen)
    return calls


# ------------------------------------
# settings
# ------------------------------------
def test_defaults(monkeypatch):
    monkeypatch.delenv('RADCALNET_OC_DATA_ROOT')
    config.load_config.cache_clear()
    cfg = config.load_config()
    assert cfg['path']['lutdata'] == str(config.default_data_root())
    assert cfg['path']['toa_lut'] == LUT_NAME
    assert cfg['zenodo']['record'] == 23394202
    assert len(cfg['settings']['aerosol_models']) == len(cfg['settings']['aerosol_combination'])
    assert config.data_root_source() == 'default folder'


def test_user_file_overrides_defaults(isolated_config, monkeypatch):
    monkeypatch.delenv('RADCALNET_OC_DATA_ROOT')
    isolated_config.write_text("path:\n  lutdata: '~/my_luts'\nprocessor:\n  ncpu: 2\n")
    config.load_config.cache_clear()
    cfg = config.load_config()
    assert cfg['path']['lutdata'] == os.path.expanduser('~/my_luts')
    assert cfg['path']['toa_lut'] == LUT_NAME            # other keys of the section kept
    assert cfg['processor']['ncpu'] == 2
    assert cfg['processor']['chunk'] == 512
    assert config.data_root_source() == 'user configuration'


def test_environment_overrides_user_file(isolated_config, tmp_path):
    isolated_config.write_text("path:\n  lutdata: '/somewhere/else'\n")
    config.load_config.cache_clear()
    assert config.load_config()['path']['lutdata'] == str(tmp_path / 'lut')
    assert config.toa_lut_path() == tmp_path / 'lut' / LUT_NAME
    assert config.data_root_source() == 'RADCALNET_OC_DATA_ROOT'


def test_missing_user_file(monkeypatch, tmp_path):
    monkeypatch.setenv('RADCALNET_OC_CONFIG', str(tmp_path / 'missing.yml'))
    config.load_config.cache_clear()
    with pytest.raises(FileNotFoundError):
        config.load_config()


def test_packaged_transmittance_lut():
    assert config.trans_lut_path().is_file()
    assert config.LUT_FILE_BACKUP.is_file()


def test_auto_download_switch(isolated_config, monkeypatch):
    assert not config.auto_download_enabled()
    monkeypatch.setenv('RADCALNET_OC_NO_DOWNLOAD', '0')
    assert config.auto_download_enabled()
    monkeypatch.delenv('RADCALNET_OC_NO_DOWNLOAD')
    assert config.auto_download_enabled()
    isolated_config.write_text('zenodo:\n  auto_download: false\n')
    config.load_config.cache_clear()
    assert not config.auto_download_enabled()


def test_init_user_config(isolated_config, monkeypatch, tmp_path):
    monkeypatch.delenv('RADCALNET_OC_DATA_ROOT')
    isolated_config.unlink()
    user_file = config.init_user_config(tmp_path / 'grsdata')
    assert user_file == isolated_config
    assert yaml.safe_load(user_file.read_text()) == {'path': {'lutdata': str(tmp_path / 'grsdata')}}
    assert config.load_config()['path']['lutdata'] == str(tmp_path / 'grsdata')
    with pytest.raises(FileExistsError):
        config.init_user_config(tmp_path / 'other')
    config.init_user_config(tmp_path / 'other', force=True)
    assert config.load_config()['path']['lutdata'] == str(tmp_path / 'other')


def test_init_user_config_default_folder(isolated_config, monkeypatch, tmp_path):
    """Without a LUT folder, the user file only holds comments: the defaults are used."""
    monkeypatch.delenv('RADCALNET_OC_DATA_ROOT')
    monkeypatch.setattr(config, 'default_data_root', lambda: tmp_path / 'default')
    isolated_config.unlink()
    text = config.init_user_config(None).read_text()
    assert yaml.safe_load(text) is None
    assert '#   lutdata: null' in text                          # packaged defaults, commented
    assert (tmp_path / 'default').is_dir()
    assert config.load_config()['path']['lutdata'] == str(tmp_path / 'default')


# ------------------------------------
# download of the TOA LUT
# ------------------------------------
def test_download(monkeypatch, tmp_path):
    calls = fake_zenodo(monkeypatch, b'lut content')
    path = config.download_toa_lut(progress=False)
    assert path == tmp_path / 'lut' / LUT_NAME
    assert path.read_bytes() == b'lut content'
    assert not path.with_name(LUT_NAME + '.part').exists()
    assert calls == ['https://zenodo.org/api/records/23394202', FILE_URL]

    # already there: no request, unless forced
    calls.clear()
    assert config.download_toa_lut(progress=False) == path
    assert calls == []
    config.download_toa_lut(force=True, progress=False)
    assert len(calls) == 2


def test_download_checksum_mismatch(monkeypatch, tmp_path):
    fake_zenodo(monkeypatch, b'corrupted', checksum='md5:' + hashlib.md5(b'lut content').hexdigest())
    with pytest.raises(OSError, match='checksum'):
        config.download_toa_lut(progress=False)
    assert list((tmp_path / 'lut').iterdir()) == []      # neither the LUT nor the .part file


def test_download_file_not_in_record(monkeypatch):
    fake_zenodo(monkeypatch, b'lut content', key='another_lut.nc')
    with pytest.raises(FileNotFoundError, match=LUT_NAME):
        config.download_toa_lut(progress=False)


def test_download_without_record(isolated_config):
    isolated_config.write_text('zenodo:\n  record: null\n')
    config.load_config.cache_clear()
    with pytest.raises(ValueError):
        config.download_toa_lut(progress=False)


def test_ensure_toa_lut_downloads(monkeypatch, tmp_path):
    monkeypatch.delenv('RADCALNET_OC_NO_DOWNLOAD')
    fake_zenodo(monkeypatch, b'lut content')
    assert config.ensure_toa_lut() == tmp_path / 'lut' / LUT_NAME
    assert (tmp_path / 'lut' / LUT_NAME).read_bytes() == b'lut content'


def test_ensure_toa_lut_download_disabled(monkeypatch):
    calls = fake_zenodo(monkeypatch, b'lut content')
    assert config.ensure_toa_lut() is None
    assert calls == []


def test_ensure_toa_lut_offline(monkeypatch):
    monkeypatch.delenv('RADCALNET_OC_NO_DOWNLOAD')

    def urlopen(url):
        raise urllib.error.URLError('no network')

    monkeypatch.setattr(config.urllib.request, 'urlopen', urlopen)
    assert config.ensure_toa_lut() is None


def test_ensure_toa_lut_present(monkeypatch, tmp_path):
    calls = fake_zenodo(monkeypatch, b'lut content')
    monkeypatch.delenv('RADCALNET_OC_NO_DOWNLOAD')
    (tmp_path / 'lut').mkdir()
    (tmp_path / 'lut' / LUT_NAME).write_bytes(b'already there')
    assert config.ensure_toa_lut() == tmp_path / 'lut' / LUT_NAME
    assert calls == []


# ------------------------------------
# radcalnet_oc_config command
# ------------------------------------
def test_command_missing_lut(capsys):
    assert config.main([]) == 1
    out, err = capsys.readouterr()
    assert 'MISSING' in out and 'radcalnet_oc_config --download' in err


def test_command_download(monkeypatch, capsys, tmp_path):
    fake_zenodo(monkeypatch, b'lut content')
    assert config.main(['--download']) == 0
    assert (tmp_path / 'lut' / LUT_NAME).is_file()
    assert 'TOA LUT          ok' in capsys.readouterr().out


def test_command_download_error(monkeypatch, capsys):
    fake_zenodo(monkeypatch, b'corrupted', checksum='md5:0')
    assert config.main(['--download']) == 1
    assert 'checksum' in capsys.readouterr().err


def test_command_init(isolated_config, monkeypatch, tmp_path):
    monkeypatch.delenv('RADCALNET_OC_DATA_ROOT')
    isolated_config.unlink()
    (tmp_path / 'grsdata').mkdir()
    (tmp_path / 'grsdata' / LUT_NAME).write_bytes(b'lut')
    assert config.main(['--init', str(tmp_path / 'grsdata')]) == 0
    assert config.main(['--init', str(tmp_path / 'grsdata')]) == 1      # already exists


@pytest.mark.skipif(not os.environ.get('RADCALNET_OC_TEST_NETWORK'),
                    reason='needs internet access: set RADCALNET_OC_TEST_NETWORK=1')
def test_zenodo_record():
    """The configured Zenodo record holds the TOA LUT of the configuration (metadata only)."""
    with urllib.request.urlopen(config.zenodo_record_url()) as response:
        record = json.load(response)
    files = {f['key']: f for f in record['files']}
    assert LUT_NAME in files
    assert files[LUT_NAME]['size'] > 5e8
    assert files[LUT_NAME]['checksum'].startswith('md5:')
