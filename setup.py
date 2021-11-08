import codecs
import os
from setuptools import setup, find_packages

MODULE_NAME = 'ner'

requires_file = 'requires.txt'
exec(f'from {MODULE_NAME} import VERSION')


################################################################################

def read_rel(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as ifh:
        return ifh.read()


def get_version(rel_path):
    for line in read_rel(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


################################################################################

# Comma separated list of names and emails
AUTHORS = ""
EMAILS = ""

# Short description
DESCRIPTION_SHORT = "..."

# Long description
with open("README.md", encoding="utf-8") as f:
    DESCRIPTION_LONG = f.read()

# Requirements
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [x for x in map(str.strip, f.read().splitlines())
                    if x and not x.startswith("#")]

# Additional (keyword) arguments
kwargs = {
    "entry_points": {
        "console_scripts": []
    }
}

####

VERSION = get_version(os.path.join(MODULE_NAME, "__init__.py"))



################################################################################

setup(
    name=MODULE_NAME,
    version=VERSION,
    description=DESCRIPTION_SHORT,
    long_description=DESCRIPTION_LONG,
    author=AUTHORS,
    author_email=EMAILS,
    url="",
    license="",
    # keywords=["kw1", "kw2"],
    install_requires=requirements,
    packages=find_packages(),
    package_data={},
    include_package_data=True,  # include MANIFEST.in data
    # ext_modules=ext_modules,
    # include_dirs=[np.get_include()],
    platforms="any",
    zip_safe=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    **kwargs
)
