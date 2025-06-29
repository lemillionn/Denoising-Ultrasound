#!/usr/bin/env bash
# Generate synthetic clean/noisy dataset
python data/generate_synthetic.py

# Train the autoencoder (uses configs/default.yaml)
python src/train.py configs/default.yaml

# Evaluate the trained model on held-out data
python src/evaluate.py
