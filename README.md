# La Morph ML Service

This is the dedicated FastAPI microservice for running DermaLens ONNX machine learning models. Moving the models from the edge (mobile app) to this service significantly reduces the app bundle size and allows for seamless model updates without requiring App Store/Play Store releases.

## Setup

1. **Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment Variables:**
```bash
cp .env.example .env
```

4. **Run the server:**
```bash
python -m app.main
```
*(Or run `uvicorn app.main:app --reload`)*

## API Endpoints

- `GET /health` : Check if the API is running and models are loaded.
- `POST /v1/analyze` : Upload an image (`multipart/form-data` with field `file` and `metadata` containing JSON face bounding boxes) to get skin analysis predictions from the ONNX models.

## Architecture

This project follows a modular structure separated into layers:
- `api/`: FastAPI route definitions.
- `core/`: Application-wide configuration and logging.
- `schemas/`: Pydantic models for validation and responses.
- `preprocessing/`: Image extraction, resizing, normalization.
- `inference/`: Executing the ONNX models.
- `postprocessing/`: Transforming model logits/outputs into structured analysis results.
- `services/`: Connecting preprocessing -> inference -> postprocessing together.
- `models/`: The actual ONNX weights (e.g., `model.fp16.onnx`, `wrinkle_unet.int8.onnx`).
