import os
from setuptools import setup, find_packages

setup(
    name='PF2nlfff',
    version='2.0.0',
    author='Guoyin Chen',
    author_email='gychen@smail.nju.edu.cn',
    description='Physics-reinforced GAN for extrapolation',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'PF2nlfff': [
            'data/*.csv', 'data/*.txt', 'data/*.json','data/*.pkl',
            'data/*.npy', 'data/*.pth', 'data/*.mat'
            ],
    },
    install_requires=[
        'numpy>=1.24.0',
        'torch>=2.4.0',
        'scipy>=1.10.0',
        'matplotlib>=3.9.0',
        'pandas>=1.4.0',
        'pyevtk>=1.6.0',
        'torchvision',
        'tqdm>=4.66.0',
    ],
    extras_require={
        'visual':[
            'vtk>=9.6.0'
            'opencv-python'
        ],
        'all':[
            'vtk>=9.6.0',
            'opencv-python'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
