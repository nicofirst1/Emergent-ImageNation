from pkg_resources import parse_requirements
from setuptools import setup, find_packages

setup(
    name='emimg',
    version='1.0',
    packages=find_packages(),
    license='MIT',
    url='https://gitlab.com/nicofirst1/Emergent-ImageNation',
    include_package_data=True,
    author=["Nicolo' Brandizzi",'Nicole Orzan'],
    install_reqs=parse_requirements('requirements.txt')

)
