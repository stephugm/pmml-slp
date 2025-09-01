# Perceptron Implementation for Iris Classification

This project implements a Perceptron algorithm to classify Iris flowers into two classes (Iris-setosa and Iris-versicolor) using their four characteristic features. The implementation uses a single-layer perceptron.

## Project Structure

- `main.py` - Main script for data loading, preprocessing, and model training/evaluation
- `perceptron.py` - Custom Perceptron class implementation
- `data.csv` - Iris dataset containing flower measurements
- `requirements.txt` - List of Python dependencies

## Requirements

The project requires Python and the following packages:
- pandas
- matplotlib
- scikit-learn

You can install all required packages using:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Iris dataset
The target classes are:
- Iris-setosa (encoded as 0)
- Iris-versicolor (encoded as 1)

## Usage

To run the classification:

```bash
python main.py
```

The script will:
1. Load and preprocess the Iris dataset
2. Split the data into training (80%) and testing (20%) sets
3. Train the perceptron model
4. Evaluate the model's performance
