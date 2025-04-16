# Excel File Processor API

## Quick Setup Guide

### 1. Setup Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
# Start the API server
python main.py

# The API will be available at http://0.0.0.0:8000
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=.
```

## API Usage

### API Endpoint

**POST** `/process-excel/`

Process an Excel file and return transformed data with concatenated columns.

### Example Request

```http
POST /process-excel/ HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "file_path": "path/to/excel_file.xlsx", 
  "required_columns": ["First Name", "Last Name", "Email"]
}
```

### Example Response

```json
{
  "success": true,
  "headers": ["First Name", "Last Name", "Email", "Concatenated"],
  "rows": [
    ["John", "Doe", "john.doe@example.com", "John - Doe - john.doe@example.com"]
  ],
  "concatenated_columns": ["First Name", "Last Name", "Email"],
  "total_rows": 1
}


### actual response in api(Method POST for API)
http://127.0.0.1:8000/process-excel 

### Payload in api


{
  "file_path": "C:/Users/Admin/Downloads/sample.xlsx",
  "required_columns": ["first_name", "last_name","address","phone"]
}

Response from api 

{
    "success": true,
    "headers": [
        "first_name",
        "last_name",
        "address",
        "phone",
        "Concatenated"
    ],
    "rows": [
        [
            "Harry",
            "Tom",
            "Noida",
            "45564",
            "Harry - Tom - Noida - 45564"
        ],
        [
            "Jane",
            "Doe",
            "Delhi",
            "55566",
            "Jane - Doe - Delhi - 55566"
        ],
        [
            "Alex",
            "Johnson",
            "UP",
            "4566",
            "Alex - Johnson - UP - 4566"
        ],
        [
            "Chris",
            "Williams",
            "Punjab",
            "556644",
            "Chris - Williams - Punjab - 556644"
        ],
        [
            "Peter",
            "Jhon",
            "Goa",
            "123468",
            "Peter - Jhon - Goa - 123468"
        ]
    ],
    "concatenated_columns": [
        "first_name",
        "last_name",
        "address",
        "phone"
    ],
    "total_rows": 5,
    "error": null
}
```


looks like passes test cases

pytest test_main.py -v --cov=main --cov-report=term
=================================================================== test session starts ===================================================================
platform win32 -- Python 3.11.9, pytest-8.3.5, pluggy-1.5.0 -- D:\project3\src\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\project3\src
plugins: anyio-4.9.0, cov-6.1.1
collected 15 items                                                                                                                                         

test_main.py::test_process_excel_success[mock_return0] PASSED                                                                                        [  6%]
test_main.py::test_process_excel_file_not_found[mock_return0] PASSED                                                                                 [ 13%]
test_main.py::test_process_excel_missing_columns[mock_return0] PASSED                                                                                [ 20%]
test_main.py::test_process_excel_general_error[mock_return0] PASSED                                                                                  [ 26%]
test_main.py::test_process_excel_unknown_error[mock_return0] PASSED                                                                                  [ 33%]
test_main.py::test_logging_setup PASSED                                                                                                              [ 40%]
test_main.py::test_log_directory_creation PASSED                                                                                                     [ 46%]
test_main.py::test_app_metadata PASSED                                                                                                               [ 53%]
test_main.py::test_app_documentation_routes PASSED                                                                                                   [ 60%]
test_main.py::test_uvicorn_run PASSED                                                                                                                [ 66%]
test_main.py::test_empty_request PASSED                                                                                                              [ 73%]
test_main.py::test_invalid_request PASSED                                                                                                            [ 80%]
test_main.py::test_invalid_method PASSED                                                                                                             [ 86%]
test_main.py::test_main_execution PASSED                                                                                                             [ 93%]
test_main.py::test_main_execution_block PASSED                                                                                                       [100%]

===================================================================== tests coverage ======================================================================
_____________________________________________________ coverage: platform win32, python 3.11.9-final-0 _____________________________________________________

Name      Stmts   Miss  Cover
-----------------------------
main.py      34      3    91%
-----------------------------
TOTAL        34      3    91%
=================================================================== 15 passed in 2.85s ==================================================================== 
## Documentation

Interactive API documentation is available at:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)


