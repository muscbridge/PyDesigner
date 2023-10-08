from setuptools import find_packages, setup

from pydesigner.info import (
    CLASSIFIERS,
    PYTHON_REQUIRES,
    REQUIRES,
    __author__,
    __description__,
    __email__,
    __license__,
    __maintainer__,
    __packagename__,
    __url__,
    __version__,
)

with open("README.rst", "r") as fh:
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
    long_description_content_type="text/x-rst",
    url=__url__,
    license=__license__,
    include_package_data=True,
    classifiers=CLASSIFIERS,
    python_requires=PYTHON_REQUIRES,
    install_requires=REQUIRES,
    entry_points={
        "console_scripts": [
            "pydesigner = designer.pydesigner:main",
        ]
    },
)
