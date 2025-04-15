from setuptools import setup, find_packages

setup(
    name='laue_indexing',
    version='0.1.0',
    packages=find_packages(include=['laue_indexing', 'laue_indexing.*']),
    include_package_data=True,
    install_requires=[
        # Add dependencies here, e.g., 'numpy', 'pyyaml', etc.
    ],
    description='A package for Laue indexing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/laue_indexing',
)