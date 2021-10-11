import os
import yaml
from setuptools import setup, find_packages
from pathlib import Path

module_name = os.path.basename(Path(os.path.realpath(__file__)).parent)
print('############ MODULE NAME: ')
print(module_name)
# requires_file = 'requires.txt'

# cwd = Path(os.path.realpath(__file__)).parent

# with open(cwd / 'settings.yaml', 'r') as f:
#   settings = yaml.load(f)

# try:
#   # Normal installation via setup.py
#   with open('requirements.txt') as f: 
#     requirements = f.read().splitlines()
# except FileNotFoundError:
#   # Happens if installed via pip of a packaged bdist (tar.gz when building wheel),
#   # because the packaged module has different structure, therefore we search for the
#   # requirements file. This works for setuptools==46.1.3
#   found_requires_path = None
#   for r, d, fs in os.walk('.', topdown=False):
#     for f in fs:
#       f_path = os.path.join(r, f)
#       if f_path.endswith(requires_file) and '.egg-info' in f_path:
#         found_requires_path = f_path
#   if (found_requires_path == None):
#     print('/!\ No requirements file found /!\ ')
#   else:
#     with open(found_requires_path) as f: 
#       requirements = f.read().splitlines()

setup(
   name=module_name,
   version='0.0.1',#settings['version'],
   description='notebook stuff',
   packages=find_packages(),
   #install_requires=requirements, # external packages as dependencies
   include_package_data=True # settings.yaml specified in MANIFEST.in
)