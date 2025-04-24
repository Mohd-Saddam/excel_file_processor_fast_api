from fastapi import FastAPI, HTTPException, status, Response, Body, Request
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
import os
import logging
from datetime import datetime
from utils.result import Result
from excel_file_process import FileProcessor, FileRequest, ProcessResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# Define static folder path
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add file handler to write logs to file
log_file_path = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Function to resolve file paths with static_path prefix
def resolve_file_path(file_path: str) -> str:
    """
    Convert static_path references to actual file paths
    
    Args:
        file_path: The file path which may contain 'static_path/' prefix
        
    Returns:
        Resolved absolute file path
    """
    if file_path and file_path.startswith("static_path/"):
        # Replace static_path with the actual static directory path
        relative_path = file_path.replace("static_path/", "", 1)
        return os.path.join(STATIC_DIR, relative_path)
    return file_path

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Excel File Processor API",
    description="API for validating and processing Excel files",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors from empty JSON payloads
    """
    # Check if this is an empty JSON payload error
    error_msg = str(exc)
    if "JSON decode error" in error_msg and "Expecting value" in error_msg:
        logger.info("Caught empty JSON payload error, handling with default values")
        # Create a default response with default values
        request_obj = FileRequest()
        # Explicitly set default values to prevent None values
        request_obj.file_path = "C:/Users/Admin/Downloads/sample.xlsx"
        request_obj.required_columns = ["address", "phone"]
        
        logger.info(f"Using default file path: {request_obj.file_path}")
        logger.info(f"Using default required columns: {request_obj.required_columns}")
        
        result = FileProcessor.process_file(request_obj)
        
        if result.is_success():
            logger.info(f"Successfully processed file with {result.data.total_rows} rows")
            # Update ProcessResponse status fields from the Result
            result.data.status_code = result.status_code.value
            result.data.status = result.status_code.phrase
            return JSONResponse(content=result.data.dict())
        else:
            error_response = ProcessResponse(
                success=False,
                error=result.error,
                status_code=result.status_code.value,
                status=result.status_code.phrase
            )
            return JSONResponse(
                content=error_response.dict(),
                status_code=result.status_code.value
            )
    
    # For other validation errors, return the standard error response
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )

# API Endpoints
@app.post(
    "/process-excel/", 
    response_model=ProcessResponse,
    tags=["Excel Processing"]
)
async def process_excel(request: FileRequest = Body(default=None), response: Response = None) -> ProcessResponse:
    """
    Process an Excel file and return transformed data.
    
    This endpoint validates an Excel file and processes its content according to
    the specified requirements. It checks for file existence, required columns,
    and transforms data into a dynamic table format with headers, rows, and metadata.
    
    Args:
        request: FileRequest object with file path and required columns (optional)
        response: FastAPI Response object to set status code
        
    Returns:
        ProcessResponse: Object containing:
            - success: Whether the operation was successful
            - status_code: HTTP status code
            - status: HTTP status description
            - headers: Column headers including concatenated column
            - rows: 2D array of row data with concatenated values
            - concatenated_columns: List of columns that were concatenated
            - total_rows: Count of rows processed
            - error: Error message if processing failed
    """
    # Set default values if request is None or empty
    if request is None:
        logger.info("Empty request received, creating a new FileRequest with default values")
        request = FileRequest()
    
    if response is None:
        response = Response()
    
    # Set default values for file_path and required_columns if not provided
    if request.file_path is None:
        request.file_path = "C:/Users/Admin/Downloads/sample.xlsx"
        logger.info(f"Using default file path: {request.file_path}")
    
    if request.required_columns is None:
        request.required_columns = ["address", "phone"]
        logger.info(f"Using default required columns: {request.required_columns}")
    
    # Resolve file path to handle static_path references
    request.file_path = resolve_file_path(request.file_path)
    
    # Log the incoming request
    logger.info(f"Processing request for file: {request.file_path}")
    
    result = FileProcessor.process_file(request)
    
    # Set the status code in the response based on the Result object
    response.status_code = result.status_code.value
    
    if result.is_success():
        logger.info(f"Successfully processed file with {result.data.total_rows} rows")
        # Update ProcessResponse status fields from the Result
        result.data.status_code = result.status_code.value
        result.data.status = result.status_code.phrase
        return result.data
    else:
        # Create an error response with the appropriate status code
        logger.warning(f"Returning error response: {result.error}")
        error_response = ProcessResponse(
            success=False,
            error=result.error,
            status_code=result.status_code.value,
            status=result.status_code.phrase
        )
        return error_response

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Excel Processor API in development mode.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
