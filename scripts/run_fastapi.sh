#!/bin/bash

HOST=${1:-"0.0.0.0"}
PORT=${2:-"8000"}

uvicorn fastapi_backend.app:app --host "${HOST}" --port "${PORT}" --reload
