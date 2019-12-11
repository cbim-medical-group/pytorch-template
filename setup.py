from setuptools import setup, find_packages

setup(
    name='PytorchTemplate',
    version='0.5.0',
    description='Annotation tool for Medical Image',
    author='Qi Chang',
    author_email='tommy.qichang@gmail.com',
    license='GPL',
    home_page='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'h5py>=2,<3',
        'pydicom>=1,<2',
        'SimpleITK>=1,<2',
        'numpy>=1,<2',
        'requests>=2,<3',
        'scipy>=1,<2',
        'torch==1.3.1',
        'pytest==5.3.1',
        'pytest-html==2.0.1',
        'pytest-cov==2.8.1',
        'nibabel==2.5.1',
        'torchvision==0.4.2',
        'pandas==0.25.3',
        'matplotlib==3.1.2',
        'scikit-image==0.16.2',
        'plotly==4.4.1',
        'jupyter==1.0.0',
        'pylama==7.7.1'
    ]
)