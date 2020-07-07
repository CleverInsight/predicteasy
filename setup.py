"""
Cognito module
"""
from codecs import open as codecs_open
from setuptools import setup, find_packages



# Get the long description from the relevant file
with codecs_open('README.rst', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


REQUIRES = [
    'tqdm',
    'numpy',            # REQ: vector algebra operations
    'scipy',
    'numpy',
    'click',            # REQ: command line interfacing
    'pandas',           # REQ: (conda) sparx.data.filter()
    'textblob',          # REQ: report generation engine
    'PyYAML',           # REQ: configuration management
    'pyfiglet',         # REQ: better cli interface  
    'PrettyTable',      # REQ: CLI based table structure
    'scikit-learn',     # REQ: simplified unity for all ML need
]


CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'Environment :: Console',
    'Intended Audience :: Developers',
    "Operating System :: OS Independent",
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python',
    'Topic :: Internet :: WWW/HTTP',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Utilities',
    'Topic :: Scientific/Engineering'

]


DOWNLOAD_URL = ""
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/CleverInsight/predicteasy",
    "Documentation": "https://predicteasy.readthedocs.io/en/latest/",
    "Source Code": "https://github.com/CleverInsight/predicteasy",
}


setup(name='predicteasy',
      version='0.0.1',
      description=u"Auto ML simplified",
      long_description="PredictEasy is an exclusive python autoML library and command line utility that helps any developer to transform raw data into a machine-learning format. ",
      classifiers=CLASSIFIERS,
      keywords=['AutoML', 'Automated Data Storyteller', 'Data Wrangler', 'Data Preprocessing',\
       'Machine Learning', 'Hot Encoder', 'Outlier Detection'],
      author=u"Bastin Robins .J",
      author_email='robin@cleverinsight.co',
      url='https://github.com/cleverinsight',
      download_url='https://github.com/CleverInsight/predicteasy/releases',
      project_urls=PROJECT_URLS,
      license='BSD',
      packages=[pkg for pkg in find_packages() if not pkg.startswith('test')],
      include_package_data=True,
      zip_safe=False,
      install_requires=REQUIRES,

      extras_require={
          'test': ['pytest'],
      },

      entry_points="""
      [console_scripts]
      cognito=cognito.scripts.cli:cli
      """)
