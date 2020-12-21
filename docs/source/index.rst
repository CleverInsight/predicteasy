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

.. code:: Python

   from predicteasy.core.data import TableFrame
   from predicteasy.core.supervised import Classifier

   df = TableFrame(pd.read_csv('data/train.csv'))
   X = df[['Sex','PassengerId', 'Pclass', 'Ticket', 'Embarked', 'Age', 'SibSp', 'Parch', 'Fare']]
   y = df['Survived']
   clf = Classifier(X, y)
   print(clf.scoring())



+----+------------+--------------------+--------------------+
| ID |   Model    |       Score        |      Elapsed       |
+----+------------+--------------------+--------------------+
| 1  | extra_tree | 1.0875861644744873 | 0.8181887043994667 |
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