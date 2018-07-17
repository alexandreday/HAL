from setuptools import setup
import sys

if sys.version_info < (3,6):
    sys.exit('Sorry, Python < 3.6 is not supported. Consider upgrading. \n See https://conda.io/docs/user-guide/tasks/manage-environments.html for a quick setup ')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name='hal-x',
      version='0.80',
      description='Clustering via hierarchical agglomerative learning',
      author='Alexandre Day',
      author_email='alexandre.day1@gmail.com',
      license='MIT',
      packages=['hal'],
      install_requires =['fitsne>=0.2.3', 'matplotlib>=2.2','scikit-learn>=0.19', 'fdc>=0.1', 'plotly>=2.5.0'],
      zip_safe=False,
      long_description=long_description,
      long_description_content_type="text/markdown",
      include_package_data=True,
      url="https://alexandreday.github.io/",
      classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
