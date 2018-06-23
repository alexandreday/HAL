from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name='hal-x',
      version='0.52',
      description='Clustering via hierarchical agglomerative learning',
      author='Alexandre Day',
      author_email='alexandre.day1@gmail.com',
      license='MIT',
      packages=['hal'],
      install_requires =['fitsne','scikit-learn'],
      zip_safe=False,
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://alexandreday.github.io/",
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
