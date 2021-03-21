from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from codecs import open
from pathlib import Path
import os


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

extensions = [
    Extension("pegasusio.cylib.funcs", ["ext_modules/fast_funcs.pyx"]),
    Extension("pegasusio.cylib.io", ["ext_modules/io_funcs.pyx"])
]

setup(
    name="pegasusio",
    use_scm_version=True,
    zip_safe=False,
    description="Pegasusio is a Python package for reading / writing single-cell genomics data",
    long_description=long_description,
    url="https://github.com/klarman-cell-observatory/pegasusio",
    author="Yiming Yang, Joshua Gould and Bo Li",
    author_email="cumulus-support@googlegroups.com",
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="single cell/nucleus genomics data reading and writing",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    setup_requires=["Cython", "setuptools_scm"],
    install_requires=[
        l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    python_requires="~=3.6",
    entry_points={"console_scripts": ["pegasusio=pegasusio.__main__:main"]},
)
