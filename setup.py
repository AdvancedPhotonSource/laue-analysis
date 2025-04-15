from setuptools import setup, find_packages

setup(
    name='laueindexing',
    version='0.1.0',
    packages=find_packages(include=['laueindexing', 'laueindexing.*']),
    include_package_data=True,
    description='A package for Laue indexing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/laue_indexing',
)