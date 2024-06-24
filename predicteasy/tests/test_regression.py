import unittest
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError
from IPython.display import IFrame
from predicteasy.endpoints.regression import RegressionAPI, ValidationError

class TestRegressionAPI(unittest.TestCase):
    
    @patch('predicteasy.endpoints.regression.requests.post')
    def test_regression_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'workflow_id': 'test_workflow_id'}
        mock_post.return_value = mock_response
        worker_url = 'http://test-worker-url'
        headers = {'Authorization': 'Bearer test_token'}
        api = RegressionAPI(worker_url, headers)
        
        result = api.regression(
            datasource_id='test_datasource_id',
            title='Test Regression',
            test_size=0.2,
            cross_val=5,
            x=['feature1', 'feature2'],
            y='target'
        )
        
        self.assertIsInstance(result, IFrame)
        self.assertIn('test_workflow_id', result.src)

        mock_post.request("Test_Validation complete")
        
    @patch('predicteasy.endpoints.regression.requests.post')
    def test_regression_validation_error(self, mock_post):
        worker_url = 'http://test-worker-url'
        headers = {'Authorization': 'Bearer fake_token'}
        api = RegressionAPI(worker_url, headers)
           
        result = api.regression(
            datasource_id='test_datasource_id',
            title='Test Regression',
            test_size=1.5,  # Invalid test_size should trigger validation error
            cross_val=5,
            x=['feature1', 'feature2'],
            y='target'
            )
            
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn('test_size', result['error']) 

    @patch('predicteasy.endpoints.regression.requests.post')
    def test_regression_api_error(self, mock_post):
        # Mock an API error scenario
        mock_response = MagicMock()
        mock_response.status_code = 500  # Simulating an internal server error
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)
        mock_post.return_value = mock_response
        
        worker_url = 'http://test-worker-url'
        headers = {'Authorization': 'Bearer test_token'}
        api = RegressionAPI(worker_url, headers)
        
        with self.assertRaises(HTTPError):
            api.regression(
                datasource_id='test_datasource_id',
                title='Test Regression',
                test_size=0.2,
                cross_val=5,
                x=['feature1', 'feature2'],
                y='target'
            )

if __name__ == '__main__':
    unittest.main()
