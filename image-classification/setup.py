from setuptools import setup


def _read(f):
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="wai.pytorchimageclass",
    description="Command-line tools for building and applying PyTorch image classification models.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/waikato-datamining/pytorch/image-classification",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='BSD-3-Clause License',
    package_dir={
        '': 'src'
    },
    packages=[
        "pic",
    ],
    version="0.0.1",
    author='Peter Reutemann',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "torch==1.6.0",
        "torchvision==0.7.0",
        "scipy==1.6.0",
        "simple-file-poller>=0.0.9",
        "python-image-complete",
    ],
    entry_points={
        "console_scripts": [
            "pic-main=pic.main:sys_main",
            "pic-predict=pic.predict:sys_main",
            "pic-poll=pic.poll:sys_main",
            "pic-export=pic.export:sys_main",
            "pic-info=pic.info:sys_main",
        ]
    }
)
