from setuptools import setup, find_packages

setup(
    name='tinytok',
    version='0.4.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'tqdm',
        'tokenizers',
    ],
    author='vxnuaj',
    description='Utility functions for processing TinyStories dataset by Eldan & Li',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vxnuaj/tinytok', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
