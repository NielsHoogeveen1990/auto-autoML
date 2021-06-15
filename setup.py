from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='autoML^2',
    keywords='',
    version='0.1',
    author='Niels Hoogeveen',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    descriptionp='A wrapper around different auto ML packages.',
    entry_points={'console_scripts': ['autoML = auto_autoML.cli:main']},
    long_description=read('README.md'),
    long_description_content_type='text/markdown'
)
