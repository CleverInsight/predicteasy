.. image:: ystatic/logo.png
    :target: http://predicteasy.readthedocs.org
    :width: 200pt


.. |pypi| image:: https://img.shields.io/pypi/v/predicteasy.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/predicteasy/

.. |conda| image:: https://img.shields.io/conda/vn/bastinrobin/predicteasy.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/bastinrobin/predicteasy

.. |travis| image:: https://travis-ci.org/CleverInsight/predicteasy.svg
   :target: https://travis-ci.org/CleverInsight/predicteasy

.. |coverall| image:: https://coveralls.io/repos/CleverInsight/predicteasy/badge.png
   :target: https://coveralls.io/r/CleverInsight/predicteasy

.. |contributors| image:: https://img.shields.io/github/contributors/Cleverinsight/predicteasy.svg?logo=github&logoColor=white
   :target: https://github.com/Cleverinsight/predicteasy/graphs/contributors/

.. |stars| image:: https://img.shields.io/github/stars/Cleverinsight/predicteasy.svg?style=social&label=Stars
   :target: https://github.com/Cleverinsight/predicteasy
   :alt: GitHub

.. |BSD| image:: https://img.shields.io/badge/License-BSD-yellow.svg
   :target: https://github.com/CleverInsight/predicteasy/blob/master/LICENSE

.. |IEEE| image:: https://img.shields.io/badge/License-BSD-yellow.svg
   :target: https://ieeexplore.ieee.org/document/9033938


.. |gitter| image:: https://img.shields.io/gitter/room/predicteasy-dev/community?color=darkviolet
   :target: https://gitter.im/predicteasy-dev/community


+----------------------+------------------------+
| Deployment           | |pypi| |conda|         |
+----------------------+------------------------+
| Build Status         | |travis|               |
+----------------------+------------------------+
| Metrics              | |coverall|             |
+----------------------+------------------------+
| GitHub               | |contributors| |stars| |
+----------------------+------------------------+
| License              | |BSD|                  |
+----------------------+------------------------+
| Community            | |gitter|               |
+----------------------+------------------------+


predicteasy : powerful autoML toolkit
==========================================

PredictEasy is an exclusive python autoML library and command line utility that helps any developer to transform raw data into a machine-learning format. 

Installation
------------

**Prerequisite**

- Python3.

Install the extension by using pip.

.. code:: bash

   $ pip install predicteasy


Quick start
==============


Classifier
==============

.. code:: python

   from predicteasy.core.data import TableFrame
   from predicteasy.core.supervised import Classifier

   df = TableFrame(pd.read_csv('data/train.csv'))
   y = df['Survived']
   X = df.drop('Survived', axis=1)
   clf = Classifier(X, y)
   print(clf.scoring(multiple=True))


Regressor
==============

.. code:: python

   from predicteasy.core.data import TableFrame
   from predicteasy.core.supervised import Regressor

   df = TableFrame(pd.read_csv('data/train.csv'))
   y = df['Age']
   X = df.drop('Age', axis=1)
   clf = Regressor(X, y)
   print(clf.scoring(multiple=True))


+----+------------+--------------------+--------------------+
| ID |   Model    |       Score        |      Elapsed       |
+----+------------+--------------------+--------------------+
| 1  | extra_tree | 1.0875861644744873 | 0.8181887043994667 |
+----+------------+--------------------+--------------------+
| 2  | dtree      | 0.8875861644744873 | 0.8181887043994667 |
+----+------------+--------------------+--------------------+
| 3  | rfc        | 0.875861644744873  | 0.8181887043994667 |
+----+------------+--------------------+--------------------+
| 4  | xgb        | 0.870861644744873  | 0.8181887043994667 |
+----+------------+--------------------+--------------------+

Contributors 
==============

.. image:: https://avatars3.githubusercontent.com/u/3523655?s=60&v=4
   :target: https://github.com/BastinRobin
.. image:: https://avatars1.githubusercontent.com/u/29769264?s=60&v=4
   :target: https://github.com/vandana-11


Core Lead
----------
* `Bastin Robins J <https://github.com/bastinrobin>`__ <robin@cleverinsight.co>

Development Leads
--------------------

* `Vandana Bhagat <https://github.com/vandana-11>`__ <vandana.bhagat@christuniversity.in>



Infrastructure support
----------------------

- We would like to thank `GitHub <https://www.github.com>`_ for providing
  us with `Github Cloud <https://www.github.com/>`_ account
  to automatically build the documentation.
  <https://github.com/CleverInsight/predicteasy>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`