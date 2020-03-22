import versioneer
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from codecs import open
import os


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

requires = [
    "Cython",
    "anndata",
    "loompy",
    "natsort",
    "numpy",
    "pandas",
    "scipy",
    "setuptools",
    "zarr"
]

extensions = [
    Extension("pegasusio.cylib.funcs", ["ext_modules/fast_funcs.pyx"]),
    Extension("pegasusio.cylib.io", ["ext_modules/io_funcs.pyx"])
]

setup(
    name="pegasusio",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    description="Pegasusio is a Python package for reading / writing single-cell genomics data",
    long_description=long_description,
    url="https://github.com/klarman-cell-observatory/pegasusio",
    author="Yiming Yang, Joshua Gould and Bo Li",
    author_email="cumulus-support@googlegroups.com",
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Framework :: Jupyter",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="single cell/nucleus genomics data reading and writing",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    setup_requires=["Cython"],
    install_requires=requires,
    python_requires="~=3.5",
)
