from setuptools import setup

setup(name='hal-x',
      version='0.5',
      description='Clustering via hierarchical agglomerative learning',
      author='Alexandre Day',
      author_email='alexandre.day1@gmail.com',
      license='MIT',
      packages=['hal'],
      install_requires =['fitsne'],
      zip_safe=False)
