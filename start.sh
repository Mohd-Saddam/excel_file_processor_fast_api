#!/bin/bash
# This script is used to start the FastAPI application with Uvicorn.
uvicorn main:app --host 0.0.0.0 #--port $PORT --reload