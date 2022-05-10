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

Update version in Docker images and create new images

* [1.6.0](docker/1.6.0/Dockerfile)
* [1.6.0_cpu](docker/1.6.0_cpu/Dockerfile)
