import argparse
from qiboosterx import QuantumTrainer, QuantumTokenizer, QuantumDataLoader

def main():
    parser = argparse.ArgumentParser(description="QiBoosterX CLI")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    data = QuantumDataLoader.load_text_data(args.data)
    tokenizer = QuantumTokenizer()
    tokenizer.build_vocab(data, preprocess=True)

    trainer = QuantumTrainer(data=data, tokenizer=tokenizer,
                             epochs=args.epochs, batch_size=args.batch_size)
    trainer.train()

if __name__ == "__main__":
    main()
