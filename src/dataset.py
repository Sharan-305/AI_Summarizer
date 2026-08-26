import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
from src.preprocessing import preprocess_dataframe
import config


class SummarizationDataset(Dataset):
    """
    PyTorch Dataset wrapper for sequence-to-sequence text summarization tasks.
    Handles tokenization, input padding, label masking, and T5 task prefixing.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str = "text", 
                 summary_col: str = "summary", max_input_length: int = 512, 
                 max_target_length: int = 150, is_t5: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.summary_col = summary_col
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.is_t5 = is_t5

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        summary = str(row[self.summary_col])

        # T5 models expect prefix 'summarize: ' before input sequence
        if self.is_t5:
            text = "summarize: " + text

        # Tokenize input document
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target summary
        targets = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )


        labels = targets["input_ids"].squeeze(0)
        # Replace padding token id with -100 so CrossEntropy loss ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }


def load_summarization_dataset(dataset_key: str = "samsum", split_sample_limit: int = None):
    """
    Flexible dataset loader supporting CNN/DailyMail, XSum, SAMSum, Multi-News, WikiHow.
    Returns cleaned pandas DataFrames for train, val, and test splits.
    """
    if dataset_key not in config.SUPPORTED_DATASETS:
        raise ValueError(f"Dataset '{dataset_key}' not supported. Options: {list(config.SUPPORTED_DATASETS.keys())}")

    dataset_info = config.SUPPORTED_DATASETS[dataset_key]
    hf_path = dataset_info["hf_path"]
    version = dataset_info["version"]
    text_col = dataset_info["text_column"]
    summary_col = dataset_info["summary_column"]

    print(f"Loading dataset '{dataset_key}' ({hf_path})...")

    try:
        if version:
            dataset_dict = load_dataset(hf_path, version)
        else:
            dataset_dict = load_dataset(hf_path)

        splits = {}
        for split_name in ["train", "validation", "test"]:
            if split_name in dataset_dict:
                ds = dataset_dict[split_name]
                if split_sample_limit and len(ds) > split_sample_limit:
                    ds = ds.select(range(split_sample_limit))
                
                df = pd.DataFrame({
                    "text": ds[text_col],
                    "summary": ds[summary_col]
                })
                splits[split_name] = preprocess_dataframe(df, "text", "summary")

        # If validation or test missing, generate fallback split
        if "validation" not in splits and "train" in splits:
            full_df = splits["train"]
            n_val = max(10, int(len(full_df) * 0.1))
            splits["validation"] = full_df.iloc[:n_val].reset_index(drop=True)
            splits["train"] = full_df.iloc[n_val:].reset_index(drop=True)

        if "test" not in splits and "validation" in splits:
            splits["test"] = splits["validation"].copy()

        return splits["train"], splits["validation"], splits["test"]

    except Exception as e:
        print(f"Warning: Failed to load dataset '{dataset_key}' from Hugging Face: {e}")
        print("Using synthetic fallback demonstration dataset...")
        return get_synthetic_dataset()


def get_synthetic_dataset():
    """
    Generates a high-quality synthetic summarization dataset for offline/demo operation.
    """
    sample_data = [
        {
            "text": "Artificial Intelligence (AI) and Machine Learning (ML) are transforming modern healthcare by enabling faster disease diagnosis, personalized treatment planning, and drug discovery. Machine learning algorithms analyze vast clinical datasets, medical imaging, and genomic data to assist physicians in early detection of conditions like cancer and diabetic retinopathy. Furthermore, AI-driven automation streamlines administrative workflows, reducing diagnostic errors and improving patient outcomes worldwide.",
            "summary": "AI and machine learning enhance healthcare through early disease diagnosis, personalized treatment, automated workflows, and accelerated drug discovery."
        },
        {
            "text": "Natural Language Processing (NLP) is a subfield of artificial intelligence focused on enabling computers to understand, interpret, and generate human language. Key application areas include text summarization, machine translation, sentiment analysis, named entity recognition, and conversational chatbots. Modern NLP heavily relies on Transformer architectures like BERT, T5, and GPT, which leverage self-attention mechanisms to capture complex contextual dependencies across long text sequences.",
            "summary": "NLP enables computers to understand human language for applications like text summarization and translation using modern Transformer models."
        },
        {
            "text": "Renewable energy sources such as solar, wind, and hydroelectric power are critical components in combating global climate change and reducing greenhouse gas emissions. Transitioning away from fossil fuels lowers atmospheric carbon dioxide levels, enhances air quality, and promotes long-term environmental sustainability. Governments and corporations are increasing investments in renewable infrastructure and energy storage solutions to ensure grid reliability and combat global warming.",
            "summary": "Renewable energy sources like solar and wind combat climate change by reducing emissions and promoting environmental sustainability."
        },
        {
            "text": "Deep learning models require significant computational resources, including High-Performance Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs). Large Language Models with billions of parameters undergo training on massive datasets over several weeks or months. Optimization techniques such as model quantization, knowledge distillation, mixed-precision training, and gradient accumulation help mitigate computational overhead and accelerate inference on edge devices.",
            "summary": "Deep learning relies on GPUs/TPUs, while optimization techniques like quantization and mixed precision reduce computational overhead for efficient inference."
        },
        {
            "text": "Quantum computing represents a paradigm shift in computer science, utilizing quantum bits or qubits that exist in superpositions of states. Unlike classical computers that operate on binary bits 0 and 1, quantum processors perform parallel calculations at exponential speeds for specific domain problems. Quantum algorithms show immense promise in cryptography, molecular modeling, material science, and complex optimization problems.",
            "summary": "Quantum computing uses qubits and superposition to perform high-speed parallel calculations for complex problems in cryptography and science."
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df = preprocess_dataframe(df, "text", "summary")
    
    # Return train, val, test
    return df, df.iloc[:2].reset_index(drop=True), df.iloc[2:].reset_index(drop=True)
