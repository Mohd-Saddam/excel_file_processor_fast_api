from fastapi import FastAPI,status, Body
import os
import logging
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your Angular app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_excel_file(file_path):
    """
    Load and parse an Excel file.
    
    Args:
        file_path (str): Path to the Excel file.
        
    Returns:
        tuple: (status_code, response_content, df) where:
            - status_code: HTTP status code
            - response_content: Response data or error message
            - df: Pandas DataFrame or None if there was an error
    """
    logger.info(f"Reading Excel file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Excel file not found: {file_path}")
        return status.HTTP_404_NOT_FOUND, {
            "error": "Excel file not found"
        }, None
    
    try:
        import pandas as pd
        df = pd.read_excel(file_path)
        
        if df.empty:
            logger.warning("Excel file is empty")
            return status.HTTP_200_OK, {
                "headers": [],
                "rows": [],
                "concatenated_columns": [],
                "total_rows": 0
            }, None
        
        return status.HTTP_200_OK, None, df
    
    except Exception as e:
        logger.exception(f"Error processing Excel file: {str(e)}")
        return status.HTTP_500_INTERNAL_SERVER_ERROR, {
            "error": f"Error processing Excel file: {str(e)}"
        }, None


def concatenate_data(df, required_columns):
    """
    Process Excel data and concatenate specified columns.
    
    Args:
        df (pandas.DataFrame): DataFrame containing Excel data
        required_columns (list): List of column names to concatenate
        
    Returns:
        dict: Processed data with concatenated values in tabular form
    """
    import pandas as pd
    
    # Dynamically get all column names directly from the Excel file
    all_headers = list(df.columns)
    logger.info(f"Excel file contains columns: {all_headers}")
    
    # Determine which columns to concatenate based on the required_columns
    concatenated_columns = [col for col in required_columns if col in all_headers]
    if not concatenated_columns:
        logger.warning("None of the specified required columns exist in the Excel file")
        # Fall back to default behavior
        if len(all_headers) >= 2:
            concatenated_columns = [all_headers[0], all_headers[1]]
        elif len(all_headers) == 1:
            concatenated_columns = [all_headers[0]]
    
    logger.info(f"Using columns for concatenation: {concatenated_columns}")
    
    # Determine which columns are not being concatenated
    non_concatenated_columns = [col for col in all_headers if col not in concatenated_columns]
    
    # Add "Concatenated" as the last header, after non-concatenated columns
    headers = non_concatenated_columns + ["Concatenated"]
    
    # Prepare rows with dynamic data - simplifying logic
    rows = []
    for _, row_data in df.iterrows():
        # Create concatenated value from the specified columns
        concat_parts = [str(row_data[col]) for col in concatenated_columns 
                       if col in row_data and pd.notna(row_data[col])]
        concat_value = " ".join(concat_parts)
        
        # Create a new row with concatenated value as the first element
        new_row = [concat_value]
        
        # Add non-concatenated values
        for col in non_concatenated_columns:
            val = row_data[col]
            if pd.notna(val):
                new_row.append(val)
            else:
                new_row.append("")
        
        rows.append(new_row)
    
    # Create the response with dynamic headers and data in tabular form
    response = {
        "headers": headers,
        "rows": rows,
        "concatenated_columns": concatenated_columns,
        "total_rows": len(rows)
    }
    
    logger.info(f"Successfully created response with {len(rows)} rows and {len(headers)} columns")
    return response


# API Endpoints
@app.get(
    "/data/", 
    tags=["Excel Processing"]
)
async def get_sample_excel_data():
    """
    Get Excel processing data with columns concatenation based on static parameters.
    
    This endpoint reads the sample.xlsx file from the static folder,
    and processes the data using predefined required columns for concatenation.
    
    Returns:
        dict: JSON response with:
            - headers: Dynamic column headers with "Concatenated" as the first header
            - rows: 2D array of row data with concatenated value as the first element
            - concatenated_columns: Columns that were concatenated
            - total_rows: Count of rows in the Excel file
    """
    logger.info("Serving dynamic Excel data from sample file")
    
    # Initialize response variables
    status_code = status.HTTP_200_OK
    response_content = {
        "headers": [],
        "rows": [],
        "concatenated_columns": [],
        "total_rows": 0
    }
    
    # Static parameters for file processing
    request_data = {
        "required_columns": ["first_name", "last_name"]
    }
    
    # Get the required columns from the static parameters
    required_columns = request_data["required_columns"]
    logger.info(f"Using static required columns for concatenation: {required_columns}")
    
    # Create a FileRequest using the sample file from static folder
    sample_file_path = os.path.join(STATIC_DIR, "excel", "sample.xlsx")
    
    # Load the Excel file
    status_code, file_response, df = load_excel_file(sample_file_path)
    
    # Process the data if file was loaded successfully
    if status_code == status.HTTP_200_OK and df is not None:
        response_content = concatenate_data(df, required_columns)
    elif file_response is not None:
        response_content = file_response
    
    # Single exit point
    if status_code != status.HTTP_200_OK:
        return JSONResponse(status_code=status_code, content=response_content)
    return response_content



# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Excel Processor API in development mode.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
