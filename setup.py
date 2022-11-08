#!/usr/bin/env python

"""The setup script."""

from pathlib import Path

from setuptools import find_packages, setup

setup(
    author="Amit Bakhru",
    author_email='bakhru@me.com',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    description="LSTM Model based stock price prediction",
    #install_requires=Path('requirements.txt').read_text().splitlines(),
    license="MIT license",
    long_description=Path('README.md').read_text(),
    include_package_data=True,
    keywords='stock_predictions',
    name='stock_predictions',
    packages=find_packages(include=['stock_predictions', 'stock_predictions.*']),
    test_suite='tests',
    url='https://github.com/abakhru/stock_prediction',
    version='0.1.0',
    zip_safe=False,
)
