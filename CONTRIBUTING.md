# Contributing to radcalnet_oc

## Development installation

```bash
git clone https://github.com/Tristanovsk/radcalnet_oc.git
cd radcalnet_oc
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

The unit tests (`tests/`) run with the light TOA LUT of the package: they hide the configuration of the
computer (empty user file, temporary LUT folder) and never download the full LUT. Two groups of tests
are skipped by default:

- the tests of the full TOA LUT, run when it is in the LUT folder of your configuration
  (`radcalnet_oc_config`), or given by `RADCALNET_OC_TEST_LUT=/path/to/toa_lut_opac_wind_v3.nc`;
- the check of the Zenodo record (metadata only), run with `RADCALNET_OC_TEST_NETWORK=1`.

The CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the tests, builds and
checks the package, runs the command line from the installed wheel and builds the documentation.

## Look-up tables

The full TOA LUT (`toa_lut_opac_wind_v3.nc`, about 900 MB) is not in the package: it is downloaded from
[Zenodo](https://doi.org/10.5281/zenodo.23394202) when the LUTs are loaded
(`radcalnet_oc.config.ensure_toa_lut`, or `radcalnet_oc_config --download`). The Zenodo record is shared
with GRS. A new LUT version means a new Zenodo record: update `zenodo: record` and `path: toa_lut` in
`radcalnet_oc/config.yml`. The other LUTs (transmittance, gases) and the light TOA LUT are in the
package (`radcalnet_oc/data`, about 14 MB in the wheel).

## Releasing on PyPI

radcalnet_oc is published on PyPI as [`radcalnet_oc`](https://pypi.org/project/radcalnet_oc/).

1. Set the version in `radcalnet_oc/__init__.py` (`__version__`, the only place: `pyproject.toml` reads
   it) and move the `[Unreleased]` entries of `CHANGELOG.md` under the new version, with the date (the
   History page of the documentation includes `CHANGELOG.md`).
2. Build from a clean checkout (old `build/` or `*.egg-info` folders would bring stale files into the
   package), and check the result:

   ```bash
   git clone --branch <branch> https://github.com/Tristanovsk/radcalnet_oc.git /tmp/radcalnet_oc_release
   cd /tmp/radcalnet_oc_release
   python -m pip install build twine
   python -m build                 # dist/radcalnet_oc-<version>.tar.gz and -py3-none-any.whl
   python -m twine check --strict dist/*
   unzip -l dist/*.whl             # about 14 MB: code, config.yml, light TOA LUT, transmittance and
                                   # gases LUTs, auxiliary data, templates; no tests, no full TOA LUT
   ```

3. Try the wheel in a new environment, tests and LUT download included:

   ```bash
   python -m venv /tmp/radcalnet_oc_venv && . /tmp/radcalnet_oc_venv/bin/activate
   pip install dist/radcalnet_oc-<version>-py3-none-any.whl pytest
   cp -r tests /tmp/radcalnet_oc_tests && (cd /tmp && pytest -q radcalnet_oc_tests)
   RADCALNET_OC_DATA_ROOT=/tmp/radcalnet_oc_lut radcalnet_oc_config --download
   radcalnet_oc --version
   ```

4. Upload to TestPyPI first, check the project page and an installation from it, then to PyPI (API
   token of the project in `~/.pypirc`, or `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<token>`):

   ```bash
   python -m twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ radcalnet_oc==<version>
   python -m twine upload dist/*
   ```

5. Tag the release (`git tag v<version> && git push --tags`).

The workflow [`.github/workflows/pypi.yml`](.github/workflows/pypi.yml) does steps 2 to 4 on GitHub
when a `v*` tag is pushed (or by hand from the Actions tab, to TestPyPI or PyPI), with PyPI trusted
publishing: no token is stored, but the GitHub repository, the workflow (`pypi.yml`) and the environment
(`pypi`, `testpypi`) must first be declared as a trusted publisher of the `radcalnet_oc` project on
pypi.org and test.pypi.org (a "pending publisher" before the first release).
