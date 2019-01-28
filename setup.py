from setuptools import setup, find_packages

setup(name='Bayesian-Outlier-Model',
      version='1.0a1',
      description='A hierarchical Bayesian model for identifying outliers in gene expression data',
      url='https://github.com/jvivian/Bayesian-Outlier-Model',
      author='John Vivian',
      author_email='jtvivian@gmail.com',
      license='MIT',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      install_requires=[
            'pymc3',
            'pandas',
            'numpy',
            'click',
            'tqdm',
            'matplotlib',
            'scipy',
            'seaborn',
            'scikit-learn'
      ],
      entry_points='''
            [console_scripts]
            outlier-model=main:cli
      ''',
      )
