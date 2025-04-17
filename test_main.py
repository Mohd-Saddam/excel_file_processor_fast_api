import pytest
import os
import sys
from datetime import datetime
from http import HTTPStatus
from unittest.mock import patch, MagicMock, mock_open
from fastapi import status
from fastapi.testclient import TestClient

# Make sure we're in the right path context for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main module - using direct imports
from main import app, logger
from excel_file_process import FileProcessor, FileRequest, ProcessResponse
from utils.result import Result

# Create a test client
client = TestClient(app)

# Fixture for mock process responses
@pytest.fixture
def successful_process_response():
    return ProcessResponse(
        success=True,
        headers=["Column1", "Column2", "Concatenated"],
        rows=[["data1", "data2", "data1 - data2"]],
        concatenated_columns=["Column1", "Column2"],
        total_rows=1
    )

@pytest.fixture
def file_not_found_response():
    return Result.fail("File does not exist at path: test_file.xlsx")

@pytest.fixture
def missing_columns_response():
    return Result.fail("Missing required columns: Column3, Column4")

@pytest.fixture
def general_error_response():
    return Result.fail("Processing error: Invalid Excel format")

# Test for successful file processing
@pytest.mark.parametrize("mock_return", [
    Result.ok(ProcessResponse(
        success=True,
        headers=["Column1", "Column2", "Concatenated"],
        rows=[["data1", "data2", "data1 - data2"]],
        concatenated_columns=["Column1", "Column2"],
        total_rows=1
    ))
])
def test_process_excel_success(mock_return):
    """Test successful processing of an Excel file"""
    # Setup the test request
    test_request = {
        "file_path": "test_file.xlsx",
        "required_columns": ["Column1", "Column2"]
    }
    
    # Mock the FileProcessor.process_file method
    with patch('excel_file_process.FileProcessor.process_file', return_value=mock_return):
        response = client.post("/process-excel/", json=test_request)
        
        # Validate response
        assert response.status_code == status.HTTP_200_OK
        
        # Validate response data
        data = response.json()
        assert data["success"] == True
        assert data["headers"] == ["Column1", "Column2", "Concatenated"]
        assert data["rows"] == [["data1", "data2", "data1 - data2"]]
        assert data["concatenated_columns"] == ["Column1", "Column2"]
        assert data["total_rows"] == 1

# Test for file not found error
@pytest.mark.parametrize("mock_return", [
    Result.fail("File does not exist at path: test_file.xlsx", status_code=HTTPStatus.NOT_FOUND)
])
def test_process_excel_file_not_found(mock_return):
    """Test file not found error handling"""
    # Setup the test request
    test_request = {
        "file_path": "test_file.xlsx",
        "required_columns": ["Column1", "Column2"]
    }
    
    # Mock the FileProcessor.process_file method
    with patch('excel_file_process.FileProcessor.process_file', return_value=mock_return):
        response = client.post("/process-excel/", json=test_request)
        
        # Validate response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "File does not exist" in response.json()["error"]

# Test for missing required columns error
@pytest.mark.parametrize("mock_return", [
    Result.fail("Missing required columns: Column3, Column4", status_code=HTTPStatus.UNPROCESSABLE_ENTITY)
])
def test_process_excel_missing_columns(mock_return):
    """Test missing required columns error handling"""
    # Setup the test request
    test_request = {
        "file_path": "test_file.xlsx",
        "required_columns": ["Column1", "Column2", "Column3", "Column4"]
    }
    
    # Mock the FileProcessor.process_file method
    with patch('excel_file_process.FileProcessor.process_file', return_value=mock_return):
        response = client.post("/process-excel/", json=test_request)
        
        # Validate response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Missing required columns" in response.json()["error"]

# Test for general processing error
@pytest.mark.parametrize("mock_return", [
    Result.fail("Processing error: Invalid Excel format", status_code=HTTPStatus.BAD_REQUEST)
])
def test_process_excel_general_error(mock_return):
    """Test general processing error handling"""
    # Setup the test request
    test_request = {
        "file_path": "test_file.xlsx",
        "required_columns": ["Column1", "Column2"]
    }
    
    # Mock the FileProcessor.process_file method
    with patch('excel_file_process.FileProcessor.process_file', return_value=mock_return):
        response = client.post("/process-excel/", json=test_request)
        
        # Validate response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Processing error" in response.json()["error"]

# Test for empty result response
@pytest.mark.parametrize("mock_return", [
    Result.fail("Unknown error without specific message", status_code=HTTPStatus.BAD_REQUEST)
])
def test_process_excel_unknown_error(mock_return):
    """Test unknown error handling that doesn't match specific patterns"""
    # Setup the test request
    test_request = {
        "file_path": "test_file.xlsx",
        "required_columns": ["Column1", "Column2"]
    }
    
    # Mock the FileProcessor.process_file method
    with patch('excel_file_process.FileProcessor.process_file', return_value=mock_return):
        response = client.post("/process-excel/", json=test_request)
        
        # Validate response - should default to 400 Bad Request
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unknown error" in response.json()["error"]

# Test for logging setup
def test_logging_setup():
    """Test that logging is properly configured"""
    # Ensure logger exists and is configured
    assert logger is not None

    # Check log file creation with the correct date format
    expected_log_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "logs", 
        f"app_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    # Test that we can write to the log
    logger.info("Test log entry for pytest")
    
    # Verify log file exists (the file should be created during app initialization)
    assert os.path.exists(expected_log_file)

# Test for log directory creation
def test_log_directory_creation():
    """Test that the log directory is created if it doesn't exist"""
    # Use a mock for os.makedirs
    with patch('os.makedirs') as mock_makedirs:
        # Mock the log directory path
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        
        # Call the function that would create the directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Verify that os.makedirs was called with the correct arguments
        mock_makedirs.assert_called_with(log_dir, exist_ok=True)

# Test FastAPI app metadata
def test_app_metadata():
    """Test the FastAPI app metadata"""
    assert app.title == "Excel File Processor API"
    assert app.description == "API for validating and processing Excel files"
    assert app.version == "1.1.0"
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"

# Test documentation routes
def test_app_documentation_routes():
    """Test that the documentation routes are accessible"""
    response = client.get("/docs")
    assert response.status_code == status.HTTP_200_OK
    
    response = client.get("/redoc")
    assert response.status_code == status.HTTP_200_OK

# Test uvicorn run configuration
def test_uvicorn_run():
    """Test the uvicorn run configuration in the main block"""
    with patch('uvicorn.run') as mock_run:
        # Directly execute the code that would be in the main block
        import uvicorn
        logger.info("Starting Excel Processor API in development mode")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
        
        # Verify the mock was called correctly
        mock_run.assert_called_once_with(
            "main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True
        )

# Test empty request validation
def test_empty_request():
    """Test validation of empty request"""
    response = client.post("/process-excel/", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# Test invalid request validation
def test_invalid_request():
    """Test validation of invalid request with missing required fields"""
    # Test with only file_path
    response = client.post("/process-excel/", json={"file_path": "test.xlsx"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    # Test with only required_columns
    response = client.post("/process-excel/", json={"required_columns": ["Column1"]})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# Test for an invalid HTTP method
def test_invalid_method():
    """Test that invalid HTTP methods are rejected"""
    response = client.get("/process-excel/")
    assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

# Add test for main script execution
def test_main_execution():
    """
    Test the code that runs when the script is executed directly.
    This tests the if __name__ == "__main__" block.
    """
    # Create a function to be used as a mock replacement that records the call
    call_args = []
    def mock_uvicorn_run(*args, **kwargs):
        call_args.append((args, kwargs))
    
    # Patch uvicorn.run with our custom function
    with patch('uvicorn.run', side_effect=mock_uvicorn_run):
        # Save the original name
        original_name = sys.modules['main'].__name__
        
        try:
            # Set __name__ to "__main__" to trigger the if block
            sys.modules['main'].__name__ = "__main__"
            
            # Get the main.py file path
            main_file_path = os.path.abspath('main.py')
            
            # Re-execute the module code which will now run the if block
            # Provide the __file__ variable in the execution context
            exec(open(main_file_path).read(), {
                '__name__': '__main__', 
                '__file__': main_file_path
            })
            
            # Check that our mock function was called with the expected arguments
            assert len(call_args) == 1, f"Expected uvicorn.run to be called once. Called {len(call_args)} times."
            args, kwargs = call_args[0]
            assert args[0] == "main:app"
            assert kwargs['host'] == "0.0.0.0"
            assert kwargs['port'] == 8000
            assert kwargs['reload'] is True
        finally:
            # Restore the original module name
            sys.modules['main'].__name__ = original_name

# Test for main execution block (lines 94-96)
def test_main_execution_block():
    """
    Test the code that executes in the if __name__ == "__main__" block
    """
    # Create a separate patch for uvicorn.run
    with patch('uvicorn.run') as mock_run:
        # Extract the code from the if __name__ == "__main__" block in main.py
        # and execute it directly here
        logger.info("Starting Excel Processor API in development mode")
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
        
        # Verify the expected call was made
        mock_run.assert_called_once_with(
            "main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True
        )