from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
import os
import logging
from datetime import datetime
from utils.result import Result
from excel_file_process import FileProcessor, FileRequest, ProcessResponse

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

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

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Excel File Processor API",
    description="API for validating and processing Excel files",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# API Endpoints
@app.post(
    "/process-excel/", 
    response_model=ProcessResponse,
    tags=["Excel Processing"]
)
async def process_excel(request: FileRequest, response: Response) -> ProcessResponse:
    """
    Process an Excel file and return transformed data.
    
    This endpoint validates an Excel file and processes its content according to
    the specified requirements. It checks for file existence, required columns,
    and transforms data into a dynamic table format with headers, rows, and metadata.
    
    Args:
        request: FileRequest object with file path and required columns
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
