PredictEasy Python SDK
======================

This is the Python SDK for PredictEasy, a platform for predictive analytics. With this SDK, you can easily interact with PredictEasy's APIs to perform various tasks such as managing datasources, performing regression analysis, classification, and clustering.

Installation
------------

To install the PredictEasy Python SDK, you can use pip:

.. code-block:: bash

    pip install predicteasy

Usage
-----

Here's a detailed tutorial on how to use each public method in the ``PredictEasyClient`` class:

1. ``list_datasources()``

    This method retrieves a list of all datasources available in PredictEasy.

    .. code-block:: python

        from predicteasy import PredictEasyClient

        # Initialize the client
        client = PredictEasyClient(auth_key="your_auth_key", auth_secret="your_auth_secret")

        # List all datasources
        datasources = client.datasource.list_datasources()
        print(datasources)

2. ``getDatasource(datasource_id)``

    This method fetches a specific datasource by its ID.

    .. code-block:: python

        # Fetch a specific datasource by ID
        datasource_id = "your_datasource_id"
        datasource = client.datasource.getDatasource(datasource_id)
        print(datasource)

3. ``createDatasource(title, description, horizontal, vertical, file_path)``

    This method creates a new datasource.

    .. code-block:: python

        # Define datasource parameters
        title = "Sample Title"
        description = "Sample Description"
        horizontal = ['CRM']
        vertical = "Telecom"
        file_path = "path/to/your/dataset.csv"

        # Create a new datasource
        new_datasource = client.datasource.createDatasource(title, description, horizontal, vertical, file_path)
        print(new_datasource)

4. ``deleteDatasource(datasource_id)``

    This method deletes a datasource by its ID.

    .. code-block:: python

        # Delete a datasource by ID
        datasource_id = "datasource_to_delete_id"
        response = client.datasource.deleteDatasource(datasource_id)
        print(response)

5. ``regression.regression(datasource_id, title, test_size, cross_val, x, y)``

    This method performs regression analysis.

    .. code-block:: python

        # Perform regression analysis
        regression_result = client.regression.regression("datasource_id", "Sales", 0.2, 2, ["feature1", "feature2"], "target")
        regression_result

6. ``classification.classify(datasource_id, title, test_size, cross_val, x, y)``

    This method performs classification.

    .. code-block:: python

        # Perform classification
        classification_result = client.classification.classify("datasource_id", "Ad Click", 0.2, 2, ["feature1", "feature2"], "target")
        classification_result

7. ``clustering.cluster(datasource_id, title, exclude, n_clusters)``

    This method performs clustering.

    .. code-block:: python

        # Perform clustering
        clustering_result = client.clustering.cluster("datasource_id", "Title", ["feature_to_exclude"], 3)
        clustering_result

Replace ``"your_auth_key"`` and ``"your_auth_secret"`` with your actual credentials from your PredictEasy Developer Profile and ``"your_datasource_id"`` with your Datasource IDs.
