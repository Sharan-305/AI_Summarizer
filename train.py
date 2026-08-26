"""
Transformer Model Fine-Tuning Script for Sharp Summarization.

Fine-tunes pretrained Seq2Seq Transformer models (BART, T5, PEGASUS)
on text summarization datasets (CNN/DailyMail, XSum, SAMSum, Multi-News, WikiHow).
"""

import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

import config
from src.dataset import load_summarization_dataset, SummarizationDataset
from src.evaluation import compute_rouge_metrics


def compute_metrics_builder(tokenizer):
    """
    Returns a compute_metrics callback function for HF Seq2SeqTrainer.
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Decode predicted tokens
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Convert -100 in labels back to pad token id for decoding
        labels = torch.tensor(labels)
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        return compute_rouge_metrics(decoded_preds, decoded_labels)

    return compute_metrics


def train(
    model_name: str = config.DEFAULT_MODEL_NAME,
    dataset_name: str = config.DEFAULT_DATASET_KEY,
    output_dir: str = config.CHECKPOINT_DIR,
    best_model_dir: str = config.BEST_MODEL_DIR,
    sample_limit: int = 1000,
    epochs: int = config.TRAIN_PARAMS["num_epochs"],
    batch_size: int = config.TRAIN_PARAMS["batch_size"],
    lr: float = config.TRAIN_PARAMS["learning_rate"]
):
    """
    Fine-tunes a pretrained Seq2Seq Transformer model (T5/BART/PEGASUS)
    with AdamW optimizer, validation monitoring, early stopping, and ROUGE evaluation.
    """
    print("=" * 65)
    print(f"       STARTING TRANSFORMER FINE-TUNING PIPELINE")
    print(f" Model           : {model_name}")
    print(f" Dataset         : {dataset_name}")
    print(f" Execution Device: {config.DEVICE.upper()}")
    print(f" Epochs          : {epochs}")
    print(f" Batch Size      : {batch_size}")
    print(f" Learning Rate   : {lr}")
    print(f" Max Samples     : {sample_limit}")
    print("=" * 65)

    # 1. Load Tokenizer & Pretrained Model Architecture
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    is_t5 = "t5" in model_name.lower()

    # 2. Load Cleaned Dataset Splits
    train_df, val_df, test_df = load_summarization_dataset(
        dataset_key=dataset_name, 
        split_sample_limit=sample_limit
    )

    print(f"Dataset Split Sizes -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 3. Create PyTorch Datasets
    train_dataset = SummarizationDataset(
        train_df, tokenizer, 
        max_input_length=config.MAX_INPUT_TOKENS, 
        max_target_length=config.MAX_TARGET_TOKENS, 
        is_t5=is_t5
    )
    
    val_dataset = SummarizationDataset(
        val_df, tokenizer, 
        max_input_length=config.MAX_INPUT_TOKENS, 
        max_target_length=config.MAX_TARGET_TOKENS, 
        is_t5=is_t5
    )

    # 4. Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=config.TRAIN_PARAMS["gradient_accumulation_steps"],
        learning_rate=lr,
        weight_decay=config.TRAIN_PARAMS["weight_decay"],
        warmup_steps=config.TRAIN_PARAMS["warmup_steps"],
        logging_steps=config.TRAIN_PARAMS["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=config.TRAIN_PARAMS["save_total_limit"],
        predict_with_generate=True,
        generation_max_length=config.MAX_TARGET_TOKENS,
        fp16=torch.cuda.is_available(),  # Enable FP16 mixed precision if GPU active
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.TRAIN_PARAMS["early_stopping_patience"])]
    )


    # 6. Execute Fine-Tuning
    print("\nExecuting model training...")
    train_result = trainer.train()

    print("\nTraining complete!")
    print(f"Global steps: {train_result.global_step} | Final loss: {train_result.training_loss:.4f}")

    # 7. Validation Evaluation
    print("\nEvaluating fine-tuned model on validation set...")
    eval_metrics = trainer.evaluate()
    print("\nValidation Metrics Summary:")
    for k, v in eval_metrics.items():
        if "rouge" in k or "loss" in k:
            print(f"  {k}: {v:.4f}")

    # 8. Save Best Fine-Tuned Weights & Tokenizer
    print(f"\nSaving best fine-tuned model weights to: {best_model_dir}")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print("Fine-tuned model successfully saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Transformer Summarization Model")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL_NAME, help="Model name or HuggingFace ID")
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET_KEY, help="Dataset key (xsum, cnn_dailymail, samsum, wikihow, multi_news)")
    parser.add_argument("--epochs", type=int, default=config.TRAIN_PARAMS["num_epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_PARAMS["batch_size"], help="Batch size per device")
    parser.add_argument("--lr", type=float, default=config.TRAIN_PARAMS["learning_rate"], help="Learning rate")
    parser.add_argument("--limit", type=int, default=500, help="Max dataset sample limit")

    args = parser.parse_args()
    
    train(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sample_limit=args.limit
    )
