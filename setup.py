from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='BayesOpt4dftu',
    version='0.0.4',
#    description='???',
    long_description=long_description,
    long_description_content_type="text/markdown",
#    author='???',
#    author_email="???",
    url='https://github.com/maituoy/BayesianOpt4dftu',
    packages=['BayesOpt4dftu'],
    install_requires=['numpy', 'ase', 'pymatgen', 'bayesian-optimization', 'pandas'],
)
