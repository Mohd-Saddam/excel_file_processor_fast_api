from fastapi import FastAPI, HTTPException, status
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
    status_code=status.HTTP_200_OK,
    tags=["Excel Processing"]
)
async def process_excel(request: FileRequest) -> ProcessResponse:
    """
    Process an Excel file and return transformed data.
    
    This endpoint validates an Excel file and processes its content according to
    the specified requirements. It checks for file existence, required columns,
    and transforms data into a dynamic table format with headers, rows, and metadata.
    
    Args:
        request: FileRequest object with file path and required columns
        
    Returns:
        ProcessResponse: Object containing:
            - headers: Column headers including concatenated column
            - rows: 2D array of row data with concatenated values
            - concatenated_columns: List of columns that were concatenated
            - total_rows: Count of rows processed
            - error: Error message if processing failed
        
    Raises:
        HTTPException: If processing fails with appropriate status code
    """
    # Log the incoming request
    logger.info(f"Processing request for file: {request.file_path}")
    
    result = FileProcessor.process_file(request)
    
    # If there was an error in processing, raise appropriate HTTP exception
    if not result.is_success():
        logger.warning(f"Returning error response: {result.error}")
        if "File does not exist" in result.error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=result.error
            )
        elif "Missing required columns" in result.error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail=result.error
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=result.error
            )
    
    logger.info(f"Successfully processed file with {result.data.total_rows} rows")
    return result.data

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Excel Processor API in development mode")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
