from setuptools import setup, find_packages
from designer.info import (
    __packagename__,
    __version__,
    __author__,
    __copyright__,
    __credits__,
    __execdir__,
    __maintainer__,
    __email__,
    __url__,
    __license__,
    __description__,
    __packages__,
    REQUIRES,
    PYTHON_REQUIRES,
    CLASSIFIERS
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name=__packagename__,
    packages=find_packages(),
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__maintainer__,
    description=__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=__url__,
    license=__license__,
    include_package_data=True,
    classifiers=CLASSIFIERS,
    python_requires=PYTHON_REQUIRES,
    install_requires=REQUIRES,
    entry_points={
            'console_scripts': [
                'pydesigner = designer.pydesigner:main',
        ]
    }
)
