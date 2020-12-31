from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='BayesOpt4dftu',
#    version='???',
#    description='???',
    long_description=long_description,
    long_description_content_type="text/markdown",
#    author='???',
#    author_email="???",
    url='https://github.com/maituoy/BayesianOpt4dftu',
    packages=['BayesOpt4dftu'],
    install_requires=['numpy', 'ase==3.19.1', 'pymatgen', 'bayesian-optimization', 'pandas', 'scikit-learn==0.21.3'],
)
