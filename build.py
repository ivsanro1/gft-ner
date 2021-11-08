'''
https://stackoverflow.com/questions/9905743/how-do-i-run-the-python-sdist-command-from-within-a-python-automated-script-wi
'''
from distutils.core import run_setup

import os

# Get module name
module_name = 'template_python_lib'

# Get version
from template_python_lib import VERSION
version = VERSION

# Get real path of this file
this_file_path = os.path.realpath(__file__)

# Get the dir
this_file_dir = os.path.dirname(this_file_path)

# Change cwd to this file dir
os.chdir(this_file_dir)

# Run setup now that the cwd points to the right place (this dir)
run_setup('setup.py', script_args=['sdist', 'bdist_wheel'])



# remote_pypi_repo_url = 'http://nexus.nlpdev2.gft.com/repository/machine-learning-platform-pypi/'
# built_package_local_abs_path = os.path.join(os.getcwd(), r'dist\{}-{}.tar.gz'.format(module_name, version))

# print('\n\n\nPackage of the module generated successfully. You can now upload it to nexus. Example on how to upload it:\n')
# print('twine upload --verbose --repository-url {} {}'.format(
#     remote_pypi_repo_url,
#     built_package_local_abs_path
# ))

# user_response = input('\nWould you like to upload the built module? ([y]/n): ')

# if user_response.lower() in ['', ' ', '\n', '\t', 'y', 'yes']:
#     from twine.commands.upload import main as twine_upload
#     twine_upload([
#         '--verbose',
#         '--repository-url',
#         remote_pypi_repo_url,
#         built_package_local_abs_path
#     ])
