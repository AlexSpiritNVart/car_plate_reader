# Car Plate Reader

This project reads license plates from video streams and can send the results to Telegram.

## Configuration

Example configuration is located in `config/config.yaml`. Update the paths to your detection and OCR model weights, Telegram credentials and stream URLs before running the application.

## Installation

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Model Weights

Place your detector weights (e.g. `.pt` file for YOLO) in a directory of your choice and specify the path in `config/config.yaml` under `path_to_detect_model`. OCR weights can be placed in the project root or another directory—update `path_to_read_model` accordingly.

## Running

After configuring the paths and credentials, start the main script:

```bash
python main.py
```

Detected plates will be processed and, if Telegram credentials are provided, sent to your specified chat.
