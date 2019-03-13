from setuptools import setup, find_packages

setup(name='Bayesian-Outlier-Model',
      version='1.0a10',
      description='A Bayesian model for identifying outliers for N-of-1 samples in gene expression data',
      url='https://github.com/jvivian/Bayesian-Outlier-Model',
      author='John Vivian',
      author_email='jtvivian@gmail.com',
      license='MIT',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      install_requires=[
          'numpy',
          'pymc3',
          'pandas',
          'tables',
          'click',
          'tqdm',
          'matplotlib',
          'scipy',
          'seaborn',
          'scikit-learn'
      ],
      entry_points=dict(console_scripts=[
          'outlier-model=outlier_model.main:cli',
          'outlier-model-mn-test=outlier_model.mn_test:cli',
          'outlier-model-weight-test=outlier_model.weight_test:cli']))
