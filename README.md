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
```

## Documentation

Interactive API documentation is available at:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)