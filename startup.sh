#!/bin/bash
# Azure App Service startup command
# Gunicorn with Uvicorn workers — handles async FastAPI correctly
pip install --quiet -r requirements.txt
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app
