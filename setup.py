import os
from setuptools import setup, find_packages

MODULE_NAME = 'ner'

requires_file = 'requires.txt'
exec(f'from {MODULE_NAME} import VERSION')

try:
  # Normal installation via setup.py
  with open('requirements.txt') as f: 
    requirements = f.read().splitlines()
except FileNotFoundError:
  # Happens if installed via pip of a packaged bdist (tar.gz when building wheel),
  # because the packaged module has different structure, therefore we search for the
  # requirements file. This works for setuptools==46.1.3
  found_requires_path = None
  for r, d, fs in os.walk('.', topdown=False):
    for f in fs:
      f_path = os.path.join(r, f)
      if f_path.endswith(requires_file) and '.egg-info' in f_path:
        found_requires_path = f_path
  if (found_requires_path == None):
    print('/!\ Requirements file not found /!\ ')
  else:
    with open(found_requires_path) as f: 
      requirements = f.read().splitlines()


setup(
   name=MODULE_NAME,
   version=VERSION,
   description='This library is for ...',
   packages=find_packages(),
   install_requires=requirements, # external packages as dependencies
   include_package_data=True # whatever is specified in MANIFEST.in
)