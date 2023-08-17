from setuptools import setup

setup(name='former',
      version='0.1',
      description='Educational implementation of self attention',
      url='http://www.peterbloem.nl/blog/transformers',
      author='Peter Bloem',
      author_email='former@peterbloem.nl',
      license='MIT',
      packages=['former'],
      install_requires=[
            'torch',
            'tqdm',
            'numpy',
            'torchtext',
            'tensorboard'
      ],
      zip_safe=False)