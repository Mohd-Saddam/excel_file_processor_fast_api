import os
import pandas as pd
import logging
import time
import uuid
from typing import List, Dict, Any
from datetime import datetime
from utils.result import Result
from pydantic import BaseModel
from typing import Optional, List
from http import HTTPStatus

# Configure logger with more structured format
logger = logging.getLogger(__name__)

class LogContext:
    """Context manager for tracking and logging operation metrics"""
    def __init__(self, operation_name: str, **kwargs):
        self.operation_name = operation_name
        self.start_time = None
        self.request_id = kwargs.get('request_id', str(uuid.uuid4())[:8])
        self.extra = kwargs
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.operation_name}", extra={"request_id": self.request_id, **self.extra})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if (exc_type):
            logger.error(
                f"Failed {self.operation_name} in {duration:.2f}s: {str(exc_val)}", 
                extra={"request_id": self.request_id, "duration": duration, **self.extra},
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            logger.info(
                f"Completed {self.operation_name} in {duration:.2f}s", 
                extra={"request_id": self.request_id, "duration": duration, **self.extra}
            )

# Response model for standardized API responses
class ProcessResponse(BaseModel):
    """
    Standardized response schema for file processing.
    
    Attributes:
        success: Whether the operation was successful
        status_code: HTTP status code of the response
        status: HTTP status description
        headers: List of column headers in the Excel file
        rows: 2D array of row data
        concatenated_columns: List of columns that were concatenated
        total_rows: Total number of rows processed
        error: Error message if unsuccessful
    """
    success: bool
    status_code: Optional[int] = 200
    status: Optional[str] = "OK"
    headers: Optional[List[str]] = None
    rows: Optional[List[List[str]]] = None
    concatenated_columns: Optional[List[str]] = None
    total_rows: Optional[int] = None
    error: Optional[str] = None

# Input validation schema
class FileRequest(BaseModel):
    """
    Schema for Excel file processing request.
    
    Attributes:
        file_path: Full path to the Excel file
        required_columns: List of column names that must exist in the file
    """
    file_path: Optional[str] = None
    required_columns: Optional[List[str]] = None

# Excel file processor with enhanced error handling
class FileProcessor:
    """
    Handles Excel file validation and processing logic.
    
    This class contains methods to:
    - Validate Excel file existence
    - Check file format
    - Verify required columns
    - Process data according to specifications
    """

    @staticmethod
    def process_file(request: FileRequest) -> Result[ProcessResponse]:
        """
        Process an Excel file according to the given requirements.
        
        Args:
            request: FileRequest object containing file path and column requirements
            
        Returns:
            Result[ProcessResponse]: Result object containing either a successful response or error
        """
        request_id = str(uuid.uuid4())[:8]
        log_context = {
            "request_id": request_id,
            "file_path": request.file_path,
            "required_columns": request.required_columns
        }
        
        # Log with structured data and request ID
        logger.info(f"Processing Excel file", extra=log_context)
        
        try:
            # Validate if the file exists and can be read as an Excel file
            with LogContext("file validation", **log_context):
                validation_result = FileProcessor._validate_file(request.file_path)
            
            # Check if validation was unsuccessful and return the error if there was one
            if not validation_result.is_success():
                logger.warning(f"File validation failed: {validation_result.error}", extra=log_context)
                return validation_result
            
            # Extract the pandas DataFrame from the successful validation result
            df = validation_result.data
            log_context["row_count"] = len(df)
            
            # Validate that all required columns exist in the DataFrame
            with LogContext("column validation", **log_context):
                column_result = FileProcessor._validate_columns(df, request.required_columns)
            
            # Check if column validation was unsuccessful and return the error if there was one
            if not column_result.is_success():
                logger.warning(f"Column validation failed: {column_result.error}", extra=log_context)
                return column_result
            
            # Process the data after validations are successful
            with LogContext("data processing", **log_context):
                process_result = FileProcessor._process_data(df, request.required_columns)
            
            if process_result.is_success():
                logger.info(
                    f"Successfully processed file with {process_result.data.total_rows} rows", 
                    extra=log_context
                )
            
            return process_result
            
        except Exception as e:
            # Log any unexpected exceptions that occur during processing
            logger.exception(f"Unexpected error during file processing", extra={**log_context, "error": str(e)})
            # Return a failure Result with the error message
            return Result.server_error(f"Processing error: {str(e)}")
    
    @staticmethod
    def _validate_file(file_path: str) -> Result[pd.DataFrame]:
        """
        Validates if the file exists and can be read as an Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Result containing either DataFrame or error message
        """
        # Check if file_path is None
        if file_path is None:
            logger.error("File path is None")
            return Result.not_found("No file path provided")
            
        # Validate if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found", extra={"file_path": file_path})
            return Result.not_found(f"File does not exist at path: {file_path}")

        # Try reading the Excel file
        try:
            logger.debug(f"Attempting to read Excel file", extra={"file_path": file_path})
            start_time = time.time()
            df = pd.read_excel(file_path)
            read_time = time.time() - start_time
            logger.info(
                f"Successfully read Excel file", 
                extra={
                    "file_path": file_path, 
                    "row_count": len(df), 
                    "column_count": len(df.columns),
                    "read_time_seconds": f"{read_time:.2f}"
                }
            )
            return Result.ok(df)
        except Exception as e:
            logger.error(
                f"Failed to read Excel file", 
                extra={
                    "file_path": file_path, 
                    "error": str(e), 
                    "error_type": type(e).__name__
                }
            )
            return Result.fail(f"Failed to read Excel file: {str(e)}", status_code=HTTPStatus.BAD_REQUEST)
    
    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> Result[bool]:
        """
        Validates that all required columns exist in the DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of column names that must exist
            
        Returns:
            Result indicating success or error with missing columns
        """
        # Filter out None values and handle the case when required_columns is None
        if required_columns is None:
            required_columns = []
        
        # Filter out None values from required_columns
        valid_required_cols = [col for col in required_columns if col is not None]
        
        # Check which required columns are missing
        missing_cols = [col for col in valid_required_cols if col not in df.columns]
        
        log_context = {
            "available_columns": list(df.columns),
            "required_columns": valid_required_cols,
            "missing_columns": missing_cols
        }
        
        if missing_cols:
            error_msg = f"Missing required columns: {', '.join(missing_cols)} from excel"
            logger.error(f"Column validation failed", extra=log_context)
            return Result.column_not_found(error_msg)
        
        logger.info(f"Column validation successful", extra=log_context)
        return Result.ok(True)
    
    @staticmethod
    def _process_data(df: pd.DataFrame, required_columns: List[str]) -> Result[ProcessResponse]:
        """
        Process the DataFrame according to requirements.
        
        Args:
            df: DataFrame with validated columns
            required_columns: List of columns to process
            
        Returns:
            Result containing ProcessResponse with processed data
        """
        log_context = {
            "total_rows": len(df),
            "required_columns": required_columns
        }
        
        # Get all available columns in the DataFrame
        available_columns = list(df.columns)
        
        # Create custom output structure with concatenated column first
        # followed by non-required columns
        
        # Check for empty DataFrame
        if df.empty:
            logger.warning("DataFrame is empty", extra=log_context)
            response = ProcessResponse(
                success=True,
                headers=[],
                rows=[],
                concatenated_columns=required_columns,
                total_rows=0,
                error="No data found",
                status_code=HTTPStatus.OK.value,
                status=HTTPStatus.OK.phrase
            )
            return Result.ok(response)
        
        # Create headers list (concatenated column + other columns)
        concat_header = "_".join(required_columns)
        other_columns = [col for col in available_columns if col not in required_columns]
        headers = [concat_header] + other_columns
        
        # Process rows
        logger.debug("Starting row concatenation process", extra=log_context)
        start_time = time.time()
        processed_rows = FileProcessor._create_rows_with_concatenation(df, required_columns, other_columns)
        processing_time = time.time() - start_time
        
        logger.info(
            f"Successfully processed data", 
            extra={
                **log_context, 
                "processed_rows": len(processed_rows),
                "processing_time_seconds": f"{processing_time:.2f}"
            }
        )
        
        response = ProcessResponse(
            success=True, 
            headers=headers,
            rows=processed_rows,
            concatenated_columns=required_columns,
            total_rows=len(processed_rows),
            status_code=HTTPStatus.OK.value,
            status=HTTPStatus.OK.phrase
        )
        return Result.ok(response)
    
    @staticmethod
    def _create_rows_with_concatenation(df: pd.DataFrame, required_columns: List[str], other_columns: List[str]) -> List[List[str]]:
        """
        Creates a 2D array of rows with custom format:
        1. First column: Values from required columns concatenated with underscore
        2. Remaining columns: Values from other columns
        
        Args:
            df: DataFrame with all columns
            required_columns: List of column names to concatenate
            other_columns: List of other columns to include after concatenated column
            
        Returns:
            List of rows in the specified format
        """
        rows = []
        for _, row in df.iterrows():
            # Create row with concatenated column first
            new_row = []
            
            # Handle required columns - concatenate with underscore
            concat_values = [str(row[col]) if pd.notna(row[col]) else "" for col in required_columns]
            concatenated = "_".join(filter(None, concat_values))
            new_row.append(concatenated)
            
            # Add values from other columns
            for col in other_columns:
                value = str(row[col]) if pd.notna(row[col]) else ""
                new_row.append(value)
            
            rows.append(new_row)
        
        return rows