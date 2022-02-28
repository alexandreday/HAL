from setuptools import setup
import sys

if sys.version_info < (3,6):
    sys.exit('Sorry, Python < 3.6 is not supported. Consider upgrading. \n See https://conda.io/docs/user-guide/tasks/manage-environments.html for a quick setup ')

with open("README.md", "r") as fh:
    long_description = fh.read()


FITSNE_MIN_VERSION = '0.2.5'
SCIKIT_MIN_VERSION = '0.19'
FDC_MIN_VERSION='1.15'
PD_MIN_VERSION='1.01'
UMAP_MIN_VERSION='0.5.2'

setup(
      name='hal-x',
      version='0.994',
      description='Clustering via hierarchical agglomerative learning',
      author='Alexandre Day',
      author_email='alexandre.day1@gmail.com',
      license='MIT',
      packages=['hal'],
      install_requires =[
        'fitsne>={0}'.format(FITSNE_MIN_VERSION),
        'scikit-learn>={0}'.format(SCIKIT_MIN_VERSION),
        'fdc>={0}'.format(FDC_MIN_VERSION),
	'pandas>={0}'.format(PD_MIN_VERSION),
	'umap-learn>={0}'.format(UMAP_MIN_VERSION)
      ],
      zip_safe=False,
      long_description=long_description,
      long_description_content_type="text/markdown",
      include_package_data=True,
      url="https://alexandreday.github.io/",
      classifiers=(
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
