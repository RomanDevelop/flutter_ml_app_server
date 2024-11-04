# flutter_ml_app_server
# Image Classifier Server

This repository contains the server-side code for an image classification application built using FastAPI and Core ML. The server is responsible for receiving images, processing them using a pre-trained machine learning model, and returning classification results.

## Features

- Upload images for classification.
- Predict the class of the uploaded image using a Core ML model.
- Returns classification results in JSON format.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- FastAPI
- Uvicorn
- PIL (Pillow)
- CoreMLTools

You can install the necessary packages using pip:

```bash
pip install fastapi uvicorn pillow coremltools
