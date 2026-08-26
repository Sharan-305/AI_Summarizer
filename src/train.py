"""
Module forwarder for train.py.
"""

from train import train, compute_metrics_builder

if __name__ == "__main__":
    import argparse
    import config

    parser = argparse.ArgumentParser(description="Fine-tune Transformer Summarization Model")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET_KEY)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_PARAMS["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_PARAMS["batch_size"])
    parser.add_argument("--lr", type=float, default=config.TRAIN_PARAMS["learning_rate"])
    parser.add_argument("--limit", type=int, default=500)

    args = parser.parse_args()
    
    train(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sample_limit=args.limit
    )
