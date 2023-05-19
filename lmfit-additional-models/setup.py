from setuptools import setup, find_packages

setup(
    name='lmfit-additional-models',
    version='1.0.2',
    author='Julian Hochhaus',
    author_email='julian.hochhaus@tu-dortmund.de',
    description='This package contains additional models for the lmfit package.',
    long_description='''# lmfit-additional-models This package contains additional models for the lmfit package. The 
    models are designed for fitting XPS spectra. This package contains background models such as Shirley, 
    Tougaard and Slope background as well as peak models such as convoluted Gaussian/Doniach-Sunjic models. In 
    addition, a convolution of a thermal distribution with a Gaussian is provided for fitting the fermi edge. The 
    models are based on the lmfit package and can be used in the same way as the models from lmfit. For more details, 
    see the [documentation](https://julian-hochhaus.github.io/lmfit-additional-models/). Feel free to contribute to 
    this package by adding new models or improving existing ones. See on GitHub: [lmfit-additional-models](
    https://github.com/Julian-Hochhaus/lmfit-additional-models)''' ,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['scipy', 'numpy', 'lmfit'],
)