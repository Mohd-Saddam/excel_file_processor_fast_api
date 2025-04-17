from typing import Generic, TypeVar, Optional, Callable, Any, Dict, Union
from http import HTTPStatus

T = TypeVar('T')  # Generic type variable
U = TypeVar('U')  # Additional type variable for map operations

class Result(Generic[T]):
    """
    A generic result class that represents the outcome of an operation.
    
    This class can be used to return either successful results with data
    or failed results with error messages in a type-safe manner.
    
    Attributes:
        success (bool): Indicates if the operation was successful
        data (Optional[T]): The result data (only present when success is True)
        error (Optional[str]): Error message (only present when success is False)
        status_code (HTTPStatus): HTTP status code (default: 200 for success, 400 for failure)
    """
    def __init__(
        self, 
        success: bool, 
        data: Optional[T] = None, 
        error: Optional[str] = None,
        status_code: Optional[Union[int, HTTPStatus]] = None
    ):
        """
        Initialize a Result object.
        
        Args:
            success (bool): Whether the operation succeeded
            data (Optional[T], optional): The data returned by a successful operation. Defaults to None.
            error (Optional[str], optional): Error message for a failed operation. Defaults to None.
            status_code (Optional[Union[int, HTTPStatus]], optional): HTTP status code. 
                Defaults to 200 for success, 400 for failure.
        """
        self.success = success
        self.data = data
        self.error = error
        
        # Set default status code based on success/failure if not provided
        if status_code is None:
            self.status_code = HTTPStatus.OK if success else HTTPStatus.BAD_REQUEST
        else:
            if isinstance(status_code, int):
                self.status_code = HTTPStatus(status_code)
            else:
                self.status_code = status_code

    @classmethod
    def ok(cls, data: T, status_code: Optional[Union[int, HTTPStatus]] = HTTPStatus.OK) -> "Result[T]":
        """
        Create a successful Result with the provided data.
        
        Args:
            data (T): The data to be wrapped in the Result
            status_code (Optional[Union[int, HTTPStatus]], optional): HTTP status code. Defaults to 200 OK.
            
        Returns:
            Result[T]: A successful Result containing the provided data
        """
        return cls(success=True, data=data, status_code=status_code)

    @classmethod
    def fail(cls, error: str, status_code: Optional[Union[int, HTTPStatus]] = HTTPStatus.BAD_REQUEST) -> "Result[T]":
        """
        Create a failed Result with the provided error message.
        
        Args:
            error (str): The error message describing the failure
            status_code (Optional[Union[int, HTTPStatus]], optional): HTTP status code. Defaults to 400 BAD_REQUEST.
            
        Returns:
            Result[T]: A failed Result containing the error message
        """
        return cls(success=False, error=error, status_code=status_code)

    @classmethod
    def not_found(cls, error: str = "Resource not found") -> "Result[T]":
        """
        Create a failed Result with NOT_FOUND status code.
        
        Args:
            error (str, optional): The error message. Defaults to "Resource not found".
            
        Returns:
            Result[T]: A failed Result with 404 status code
        """
        return cls(success=False, error=error, status_code=HTTPStatus.NOT_FOUND)
        
    @classmethod
    def column_not_found(cls, error: str = "Column not found in Excel file") -> "Result[T]":
        """
        Create a failed Result specific for Excel columns not found with NOT_FOUND status code.
        
        Args:
            error (str, optional): The error message about missing columns. Defaults to "Column not found in Excel file".
            
        Returns:
            Result[T]: A failed Result with 404 status code
        """
        return cls(success=False, error=error, status_code=HTTPStatus.NOT_FOUND)

    @classmethod
    def invalid_input(cls, error: str = "Invalid input data") -> "Result[T]":
        """
        Create a failed Result with BAD_REQUEST status code.
        
        Args:
            error (str, optional): The error message. Defaults to "Invalid input data".
            
        Returns:
            Result[T]: A failed Result with 400 status code
        """
        return cls(success=False, error=error, status_code=HTTPStatus.BAD_REQUEST)

    @classmethod
    def server_error(cls, error: str = "Internal server error") -> "Result[T]":
        """
        Create a failed Result with INTERNAL_SERVER_ERROR status code.
        
        Args:
            error (str, optional): The error message. Defaults to "Internal server error".
            
        Returns:
            Result[T]: A failed Result with 500 status code
        """
        return cls(success=False, error=error, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    def is_success(self) -> bool:
        """
        Check if the Result represents a successful operation.
        
        Returns:
            bool: True if the Result is successful, False otherwise
        """
        return self.success
        
    def is_failure(self) -> bool:
        """
        Check if the Result represents a failed operation.
        
        Returns:
            bool: True if the Result is a failure, False otherwise
        """
        return not self.success
        
    def unwrap(self, default: Optional[T] = None) -> Optional[T]:
        """
        Safely access the data value with an optional default value.
        
        Args:
            default (Optional[T], optional): Value to return if the Result is a failure. Defaults to None.
            
        Returns:
            Optional[T]: The data value if successful, otherwise the default value
        """
        return self.data if self.is_success() else default
        
    def unwrap_or_raise(self) -> T:
        """
        Get the data value or raise an exception if the Result is a failure.
        
        Raises:
            ValueError: If the Result is a failure, with the error message
            
        Returns:
            T: The data value
        """
        if not self.is_success():
            raise ValueError(self.error or "Operation failed")
        return self.data  # type: ignore
        
    def map(self, fn: Callable[[T], U]) -> "Result[U]":
        """
        Apply a function to the data if the Result is successful.
        
        Args:
            fn (Callable[[T], U]): Function to apply to the data
            
        Returns:
            Result[U]: A new Result with the transformed data or the original error
        """
        if self.is_success():
            return Result.ok(fn(self.data), status_code=self.status_code)  # type: ignore
        return Result.fail(self.error or "", status_code=self.status_code)  # type: ignore
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Result to a dictionary suitable for API responses.
        
        Returns:
            Dict[str, Any]: Dictionary containing status, status_code, data/error
        """
        response = {
            "success": self.success,
            "status_code": self.status_code.value,
            "status": self.status_code.phrase
        }
        
        if self.is_success():
            response["data"] = self.data
        else:
            response["error"] = self.error
            
        return response
        
    def __str__(self) -> str:
        """
        Get a string representation of the Result.
        
        Returns:
            str: String representation of the Result
        """
        status_info = f"{self.status_code.value} {self.status_code.phrase}"
        if self.is_success():
            data_repr = str(self.data)
            # Truncate long data representations
            if len(data_repr) > 100:
                data_repr = f"{data_repr[:97]}..."
            return f"Success ({status_info}): {data_repr}"
        return f"Failure ({status_info}): {self.error}"
        
    def __repr__(self) -> str:
        """
        Get a detailed string representation of the Result.
        
        Returns:
            str: Detailed string representation of the Result
        """
        return f"Result(success={self.success}, status_code={self.status_code!r}, data={self.data!r}, error={self.error!r})"
        
    def and_then(self, fn: Callable[[T], "Result[U]"]) -> "Result[U]":
        """
        Chain operations that return Result objects.
        
        This method is useful for composing multiple operations that can fail.
        If this Result is a failure, it short-circuits and returns itself.
        If it's a success, it applies the function to the data and returns the new Result.
        
        Args:
            fn (Callable[[T], Result[U]]): Function that takes the success data and returns a new Result
            
        Returns:
            Result[U]: Either the original failure or the new Result from the function
        """
        if not self.is_success():
            # Need to cast to make type checker happy
            return Result.fail(self.error or "", status_code=self.status_code)  # type: ignore
        return fn(self.data)  # type: ignore
        
    def on_success(self, fn: Callable[[T], None]) -> "Result[T]":
        """
        Execute a side effect function if the Result is successful.
        
        Args:
            fn (Callable[[T], None]): Function to execute with the data
            
        Returns:
            Result[T]: The original Result, unchanged
        """
        if self.is_success():
            fn(self.data)  # type: ignore
        return self
        
    def on_failure(self, fn: Callable[[str], None]) -> "Result[T]":
        """
        Execute a side effect function if the Result is a failure.
        
        Args:
            fn (Callable[[str], None]): Function to execute with the error message
            
        Returns:
            Result[T]: The original Result, unchanged
        """
        if not self.is_success():
            fn(self.error or "")  # type: ignore
        return self