# PyPi

Preparation:

* increment version in `setup.py`
* add new changelog section in `CHANGES.rst`
* commit/push all changes

Commands for releasing on pypi.org (requires twine >= 1.8.0):

```commandline
find -name "*~" -delete
rm dist/*
./venv/bin/python setup.py clean
./venv/bin/python setup.py sdist
./venv/bin/twine upload dist/*
```

# Debian

Generate Debian package with the following commands (requires `python3-all` and `python3-stdeb`):

```commandline
rm -Rf deb_dist/*
python3 setup.py --command-packages=stdeb.command bdist_deb
```

# Github

Steps:

* start new release (version: `vX.Y.Z`)
* enter release notes, i.e., significant changes since last release
* upload `pytorch-image-classification-X.Y.Z.tar.gz` previously generated with `setyp.py`
* publish


