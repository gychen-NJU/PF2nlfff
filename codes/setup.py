from setuptools import setup, find_packages

project_name = "PF2nlfff"
required = ['numpy', 'scipy', 'matplotlib', 'torch']
version = '1.0.5'

setup(
    name=project_name,
    version=version,
    author='G.Y.Chen',
    author_email='gychen@smail.nju.edu.cn',
    description='Neural network to generate NLFFF from a Potential Field',
    url='https://github.com/gychen-NJU',
    keywords="NLFFF, neural network",
    license='MIT',
    python_requires=">=3.7",
    install_requires=required,
    packages=find_packages(),
)