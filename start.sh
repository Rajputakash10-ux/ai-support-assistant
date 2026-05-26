#!/usr/bin/env bash
set -e

echo "=== Testing Python imports ==="
python3 -c "
import sys
print('Python:', sys.version)

print('Testing fastapi...')
from fastapi import FastAPI
print('OK')

print('Testing backend.main...')
from backend.main import app
print('OK')

print('Testing uvicorn...')
import uvicorn
print('OK')

print('All imports successful!')
"

echo "=== Starting uvicorn ==="
exec python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level debug
