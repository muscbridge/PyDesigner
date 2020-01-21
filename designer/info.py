import inspect, os

__execdir__ = os.path.basename(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

__packagename__ = 'PyDesigner'
__version__='0.2'
__author__ = 'PyDesigner developers'
__copyright__ = 'Copyright 2020, PyDesigner developers, MUSC Advanced Image Analysis (MAMA)'
__credits__ = [
    'Siddhartha Dhiman',
    'Joshua Teves',
    'Kayti Keith',
    'Benjamin Ades-Aron',
    'Jelle Veraart',
    'Vitria Adisetiyo',
    'Els Fieremans',
    'Jens Jensen'
]
__maintainer__ = 'Siddhartha Dhiman'
__email__ = 'mama@musc.edu'
__url__ = 'https://github.com/m-ama/PyDesigner'
__license__='MPL 2.0'
__description__ = ('Python Port of NYU\'s Designer pipeline for DMRI '
                'processing')
# Gets folder name where this file resides
__execdir__ = os.path.basename(
    os.path.dirname(
        os.path.abspath(
            inspect.getfile(
                inspect.currentframe()
                )
            )
        )
    )

# PyPi package requirements
REQUIRES = [
    'numpy',
    'scipy',
    'matplotlib',
    'py-cpuinfo',
    'joblib',
    'tqdm',
    'multiprocess',
    'nibabel',
    'cvxpy'
]

# Python version requirements
PYTHON_REQUIRES = ">=3.7"

# Package classifiers
CLASSIFIERS = [
    'Programming Language :: Python :: 3.7',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Bio-Informatics'
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics'
]
