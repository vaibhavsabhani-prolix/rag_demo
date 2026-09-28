#!/bin/bash
# FastAPI backend for the React frontend (frontend/). Serves on http://localhost:8000
# and also serves frontend/dist when it has been built with `npm run build`.
VENV=${VENV:-.venv}
PYTHONPATH=. "$VENV"/bin/uvicorn app.api.main:app --host 0.0.0.0 --port 8000
