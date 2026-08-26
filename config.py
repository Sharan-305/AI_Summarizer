import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "best_model")

# Create necessary directories
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# Device Configuration
DEVICE_OVERRIDE = os.getenv("PREFERRED_DEVICE", "auto")
if DEVICE_OVERRIDE == "cuda" and torch.cuda.is_available():
    DEVICE = "cuda"
elif DEVICE_OVERRIDE == "cpu":
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Supported Pretrained Transformer Models for Sharp Summarization
SUPPORTED_MODELS = {
    "sshleifer/distilbart-cnn-12-6": {
        "name": "sshleifer/distilbart-cnn-12-6",
        "type": "bart",
        "max_input_length": 1024,
        "max_target_length": 142,
        "description": "Distilled BART CNN model - extremely fast, highly accurate abstractive summaries."
    },
    "t5-small": {
        "name": "t5-small",
        "type": "t5",
        "max_input_length": 512,
        "max_target_length": 150,
        "description": "Lightweight T5 model, efficient CPU training & inference."
    },
    "facebook/bart-base": {
        "name": "facebook/bart-base",
        "type": "bart",
        "max_input_length": 1024,
        "max_target_length": 142,
        "description": "Standard BART architecture tuned for articles and reports."
    },
    "google/pegasus-xsum": {
        "name": "google/pegasus-xsum",
        "type": "pegasus",
        "max_input_length": 512,
        "max_target_length": 64,
        "description": "PEGASUS XSum - trained specifically for extreme concise summarization."
    }
}

# Default Selected Model
DEFAULT_MODEL_KEY = "t5-small"
DEFAULT_MODEL_NAME = SUPPORTED_MODELS[DEFAULT_MODEL_KEY]["name"]


# Supported Datasets
SUPPORTED_DATASETS = {
    "cnn_dailymail": {
        "hf_path": "cnn_dailymail",
        "version": "3.0.0",
        "text_column": "article",
        "summary_column": "highlights",
        "description": "Long news articles with multi-sentence highlight summaries."
    },
    "xsum": {
        "hf_path": "xsum",
        "version": None,
        "text_column": "document",
        "summary_column": "summary",
        "description": "Extreme Summarization dataset for sharp single-sentence summaries."
    },
    "samsum": {
        "hf_path": "samsum",
        "version": None,
        "text_column": "dialogue",
        "summary_column": "summary",
        "description": "Messenger-style dialogue summarization dataset."
    },
    "multi_news": {
        "hf_path": "multi_news",
        "version": None,
        "text_column": "document",
        "summary_column": "summary",
        "description": "Multi-document news summarization dataset."
    },
    "wikihow": {
        "hf_path": "wikihow",
        "version": "all",
        "text_column": "text",
        "summary_column": "headline",
        "description": "How-to step-by-step article summarization."
    }
}

# Default Dataset Configuration
DEFAULT_DATASET_KEY = "xsum"

# Hierarchical Long-Document Processing Parameters
CHUNK_SIZE = 400            # Words per chunk
CHUNK_OVERLAP = 50          # Overlap words between chunks
MAX_INPUT_TOKENS = 512      # Token limit per transformer pass
MAX_TARGET_TOKENS = 150     # Target summary token length

# Summary Mode Specifications required by user prompt
SUMMARY_MODES = {
    "Ultra Short": {
        "num_sentences": (1, 2),
        "max_words": 30,
        "min_words": 10,
        "description": "1–2 sentences, only central idea"
    },
    "Short": {
        "num_sentences": (2, 4),
        "max_words": 60,
        "min_words": 25,
        "description": "2–4 sentences, main facts and conclusion"
    },
    "Brief": {
        "num_sentences": (4, 6),
        "max_words": 120,
        "min_words": 50,
        "description": "4–6 sentences, important details without unnecessary explanation"
    }
}

# Generation Hyperparameters for Sharp Concise Summaries
SHARP_GEN_PARAMS = {
    "num_beams": 4,
    "length_penalty": 0.6,      # Penalize long outputs to encourage sharp conciseness
    "no_repeat_ngram_size": 3,  # Prevent 3-gram repetitions
    "early_stopping": True
}

# Fine-Tuning Training Hyperparameters
TRAIN_PARAMS = {
    "batch_size": 4,
    "eval_batch_size": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_epochs": 3,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 2,
    "save_total_limit": 2,
    "logging_steps": 50,
    "eval_steps": 200,
    "save_steps": 200,
    "early_stopping_patience": 3,
    "seed": 42
}
