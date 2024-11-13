from pathlib import Path
from setuptools import find_packages, setup

VERSION = '0.0.2'
PACKAGE_NAME = 'GNT'
AUTHOR = 'Camilo Andres Rodriguez Reyes, Ariel Camilo Sanchez Lopez'
AUTHOR_EMAIL = 'carodriguezreyes@ucundinamarca.edu.co,arielcsanchez@ucundinamarca.edu.co'
URL = 'https://github.com/CAMILORR23/GNTLibrary'

LICENSE = 'MIT'
DESCRIPTION = 'Una librería para implementar algoritmos genéticos en proyectos de optimización, ideal para resolver problemas de maximización y minimización de funciones mediante procesos evolutivos.'
LONG_DESCRIPTION = Path("README.md").read_text(encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'numpy',
    'matplotlib'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)
