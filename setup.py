from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='BayesOpt4dftu',
    version='0.1.4',
#    description='???',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Maituo Yu',
#    author_email="???",
    url='https://github.com/maituoy/BayesianOpt4dftu',
    packages=['BayesOpt4dftu'],
    install_requires=['numpy', 'ase==3.22.0', 'pymatgen==2022.0.16', 'bayesian-optimization==1.2.0', 'pandas','vaspvis==1.2.2'],
)
