import os
import pandas as pd
import logging
from typing import List
from datetime import datetime
from utils.result import Result
from pydantic import BaseModel
from typing import Optional, List

# Get logger
logger = logging.getLogger(__name__)

# Response model for standardized API responses
class ProcessResponse(BaseModel):
    """
    Standardized response schema for file processing.
    
    Attributes:
        success: Whether the operation was successful
        headers: List of column headers in the Excel file
        rows: 2D array of row data
        concatenated_columns: List of columns that were concatenated
        total_rows: Total number of rows processed
        error: Error message if unsuccessful
    """
    success: bool
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
    file_path: str
    required_columns: List[str]

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
        # Log the beginning of file processing with the file path
        logger.info(f"Processing file: {request.file_path}")
        
        try:
            # Validate if the file exists and can be read as an Excel file
            # This calls the _validate_file method which returns a Result object
            validation_result = FileProcessor._validate_file(request.file_path)
            
            # Check if validation was unsuccessful and return the error if there was one
            if not validation_result.is_success():
                return validation_result
            
            # Extract the pandas DataFrame from the successful validation result
            df = validation_result.data
            
            # Validate that all required columns exist in the DataFrame
            # This calls the _validate_columns method which returns a Result object
            column_result = FileProcessor._validate_columns(df, request.required_columns)
            
            # Check if column validation was unsuccessful and return the error if there was one
            if not column_result.is_success():
                return column_result
            
            # Process the data after validations are successful
            # This calls the _process_data method which returns a Result object with ProcessResponse
            return FileProcessor._process_data(df, request.required_columns)
            
        except Exception as e:
            # Log any unexpected exceptions that occur during processing
            logger.exception(f"Unexpected error during file processing: {str(e)}")
            # Return a failure Result with the error message
            return Result.fail(f"Processing error: {str(e)}")
    
    @staticmethod
    def _validate_file(file_path: str) -> Result[pd.DataFrame]:
        """
        Validates if the file exists and can be read as an Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Result containing either DataFrame or error message
        """
        # Validate if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return Result.fail(f"File does not exist at path: {file_path}")

        # Try reading the Excel file
        try:
            logger.info(f"Reading Excel file: {file_path}")
            df = pd.read_excel(file_path)
            logger.info(f"Successfully read file with {len(df)} rows")
            return Result.ok(df)
        except Exception as e:
            logger.error(f"Failed to read Excel file: {str(e)}")
            return Result.fail(f"Failed to read Excel file: {str(e)}")
    
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
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {', '.join(missing_cols)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
        
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
        # Using only the requested columns
        df_subset = df[required_columns]
        
        # Check for empty DataFrame
        if df_subset.empty:
            logger.warning("DataFrame is empty after selecting required columns")
            return Result.ok(ProcessResponse(
                success=True,
                headers=[],
                rows=[],
                concatenated_columns=[],
                total_rows=0,
                error="No data found after applying column filter"
            ))
        
        # Create headers list (original columns + concatenated)
        headers = required_columns.copy()
        headers.append("Concatenated")
        
        # Process rows
        processed_rows = FileProcessor._create_rows_with_concatenation(df_subset)
        
        logger.info(f"Successfully processed {len(processed_rows)} rows")
        return Result.ok(ProcessResponse(
            success=True, 
            headers=headers,
            rows=processed_rows,
            concatenated_columns=required_columns,
            total_rows=len(processed_rows)
        ))
    
    @staticmethod
    def _create_rows_with_concatenation(df_subset: pd.DataFrame) -> List[List[str]]:
        """
        Creates a 2D array of rows with concatenated values.
        
        Args:
            df_subset: DataFrame with only the required columns
            
        Returns:
            List of rows where each row contains original values plus concatenated value
        """
        rows = []
        for _, row in df_subset.iterrows():
            # Handle NaN values properly
            values = row.fillna("").astype(str).tolist()
            # Create concatenated value
            concatenated = " - ".join(filter(None, values))
            # Append concatenated value to the row
            row_data = values + [concatenated]
            rows.append(row_data)
        
        return rows