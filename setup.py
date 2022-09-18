from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='HuBMAP + HPA - Hacking the Human Body Kaggle competition. The goal of this competition is to identify the locations of each functional tissue unit (FTU) in biopsy slides from several different organs. The underlying data includes imagery from different sources prepared with different protocols at a variety of resolutions, reflecting typical challenges for working with medical data.',
    author='Alex Rodriguez',
    license='MIT',
)
