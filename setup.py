from setuptools import setup

setup(
    name='ccns',
    version='0.1',
    packages=['ccns',
              'ccns.writers',
              'ccns.reduction',
              'ccns.logger',
              'ccns.visualization',
              'ccns.visualization.jupyter'],
    url='https://github.com/burkeds/cnbl',
    license='',
    author='burkeds',
    author_email='burkeds@mcmaster.ca',
    description='SANS data reduction and analysis package for instruments at the Canadian Neutron Beam Laboratories @ '
                'McMaster University.'
)
