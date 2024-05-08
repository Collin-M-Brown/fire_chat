import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

def read_requirements():
    with open(r'C:\Users\cbrow\crbw\Git\fire_chat\requirements.txt', 'r') as req:
        content = req.read()
        requirements = content.split('\n')
        requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]
    return requirements

setup(
    name='fire_chat',
    version='0.1.0',
    author='Collin M Brown',
    author_email='collin.brown.m@gmail.com',
    package_dir={"": "src"},  # Set the package directory to 'src'
    packages=find_packages(where="src"),  # Look for packages in 'src'
    url='http://pypi.python.org/pypi/MyPackageName/',
    license='LICENSE.txt',
    description='Modules for making chat with fireworks easy',
    long_description=open(os.path.join(here, 'README.md')).read(),
    long_description_content_type='text/markdown',
    install_requires=read_requirements(),
)