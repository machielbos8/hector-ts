import os
import sys
import subprocess
from pathlib import Path
import setuptools
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np


def _user_fftw_lib():
    """Return ~/.local/lib if it contains a user-compiled libfftw3, else None."""
    p = Path.home() / '.local' / 'lib'
    if (p / 'libfftw3.so').exists() or (p / 'libfftw3.dylib').exists():
        return str(p)
    return None


def _fftw_include_dirs():
    """Return a list of include directories that contain fftw3.h, portably."""
    # 1. Honour an explicit override (useful on Windows or exotic installs).
    #    Accept both FFTW_DIR=/path/to/include (direct) and FFTW_DIR=/path/to/prefix
    #    (vcpkg layout: prefix/include/fftw3.h).
    fftw_dir = os.environ.get('FFTW_DIR')
    if fftw_dir:
        inc_sub = os.path.join(fftw_dir, 'include')
        if os.path.isfile(os.path.join(inc_sub, 'fftw3.h')):
            return [inc_sub]
        return [fftw_dir]

    # 2. Prefer a user-local FFTW (e.g., compiled with AVX-512).
    user_inc = str(Path.home() / '.local' / 'include')
    if os.path.isfile(os.path.join(user_inc, 'fftw3.h')):
        return [user_inc]

    # 3. Ask pkg-config (works on Linux and macOS with Homebrew/MacPorts).
    try:
        out = subprocess.check_output(
            ['pkg-config', '--cflags-only-I', 'fftw3'],
            stderr=subprocess.DEVNULL, text=True,
        )
        dirs = [tok[2:] for tok in out.split() if tok.startswith('-I')]
        if dirs:
            return dirs
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 4. Search common platform-specific locations.
    candidates = [
        '/usr/include',
        '/usr/local/include',
    ]
    if sys.platform == 'darwin':
        candidates += [
            '/opt/homebrew/include',   # Apple Silicon Homebrew
            '/usr/local/include',      # Intel Homebrew / MacPorts
        ]
    elif sys.platform == 'win32':
        candidates += [
            r'C:\vcpkg\installed\x64-windows\include',
            r'C:\fftw3',
            r'C:\Program Files\fftw3\include',
            r'C:\Program Files (x86)\fftw3\include',
        ]

    found = [d for d in candidates if os.path.isfile(os.path.join(d, 'fftw3.h'))]
    if found:
        return found

    raise RuntimeError(
        "fftw3.h not found. Install FFTW3 (e.g. 'brew install fftw', "
        "'apt install libfftw3-dev', or 'conda install -c conda-forge fftw') "
        "or set the FFTW_DIR environment variable to its include directory."
    )


def _fftw_lib_dirs():
    """Return library directories containing libfftw3, from pkg-config or env."""
    # User-local install (e.g., compiled with AVX-512) takes priority.
    user_lib = _user_fftw_lib()
    if user_lib:
        return [user_lib]
    # Ask pkg-config for -L paths (works on Linux and macOS with Homebrew).
    try:
        out = subprocess.check_output(
            ['pkg-config', '--libs-only-L', 'fftw3'],
            stderr=subprocess.DEVNULL, text=True,
        )
        dirs = [tok[2:] for tok in out.split() if tok.startswith('-L')]
        if dirs:
            return dirs
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return []


def _fftw_ext_kwargs():
    """Return Extension kwargs for Cython modules that link against FFTW3."""
    inc = [np.get_include()] + _fftw_include_dirs()
    lib_dirs = _fftw_lib_dirs()
    kwargs = {
        'include_dirs': inc,
        'libraries': ['fftw3'],   # required on all platforms
    }
    if lib_dirs:
        kwargs['library_dirs'] = lib_dirs
        # Embed rpath on Linux so the .so resolves without LD_LIBRARY_PATH.
        if sys.platform != 'win32' and sys.platform != 'darwin':
            kwargs['runtime_library_dirs'] = lib_dirs
    if sys.platform == 'win32':
        # On Windows, derive lib dir from the include dir (FFTW_DIR points to
        # …/include, sibling …/lib holds fftw3.lib; vcpkg layout as fallback).
        inc_dirs = _fftw_include_dirs()
        if inc_dirs:
            win_lib = os.path.join(os.path.dirname(inc_dirs[0]), 'lib')
        else:
            win_lib = r'C:\vcpkg\installed\x64-windows\lib'
        kwargs['library_dirs'] = [win_lib]
        # MSVC in C mode detects _Complex_I from <complex.h> and then tries
        # "typedef _Complex double fftw_complex" which MSVC does not support.
        # FFTW_NO_Complex forces the double[2] fallback, matching the pyx
        # ctypedef double fftw_complex[2] declaration.
        kwargs['define_macros'] = [('FFTW_NO_Complex', None)]
    return kwargs

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hector-ts",
    version="3.0.4",
    author="Machiel Bos",
    author_email="machielbos@protonmail.com",
    description="A collection of programs to analyse geodetic time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/machielsimonbos/hector-ts",
    project_urls={
        "Bug Tracker": "https://gitlab.com/machielsimonbos/hector-ts/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: Free for non-commercial use",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    keywords=[
        "geodesy", "GNSS", "GPS", "time series", "trend estimation",
        "noise analysis", "maximum likelihood", "power-law noise",
        "offset detection", "Toeplitz",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={"hector": [
        "*.pyx", "*.pxd",
        "docs/*",
        "examples/*",
        "examples/*/*",
        "examples/*/*/*",
    ]},
    python_requires=">=3.6",
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'mpmath',
        'cython',
        'netCDF4',
    ],
    ext_modules=cythonize(
        [
            "src/hector/_epoch_scan_gaps.pyx",
            "src/hector/_epoch_scan_nogap.pyx",
            "src/hector/_gap_correction.pyx",
            "src/hector/_ggm.pyx",
            "src/hector/_levinson.pyx",
            Extension(
                "hector._schur_gsa",
                sources=["src/hector/_schur_gsa.pyx"],
                **_fftw_ext_kwargs(),
            ),
        ],
        language_level="3",
    ),
    include_dirs=[np.get_include()],
    entry_points ={
        'console_scripts': [
            'estimatespectrum = hector.estimatespectrum:main',
            'modelspectrum = hector.modelspectrum:main',
            'estimatetrend = hector.estimatetrend:main',
            'estimate_all_trends = hector.estimate_all_trends:main',
            'removeoutliers = hector.removeoutliers:main',
            'findoffsets = hector.findoffsets:main',
            'find_all_offsets = hector.find_all_offsets:main',
            'simulatenoise = hector.simulatenoise:main',
            'mjd2date = hector.mjd2date:main',
            'date2mjd = hector.date2mjd:main',
            'convert_rlrdata2mom = hector.convert_rlrdata2mom:main',
            'convert_tenv2netcdf = hector.convert_tenv2netcdf:main',
            'predicttrenderror = hector.predicttrenderror:main',
            'ncfgen = hector.ncfgen:main',
            'ncfdump = hector.ncfdump:main',
            'plot_ts = hector.plot_ts:main',
            'hector-examples = hector.hector_examples:main',
        ],
    }
)
