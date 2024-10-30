# Genetic Algorithm for Classifier Optimization

This project implements a genetic algorithm designed to optimize classifier weights for a text classification task. The algorithm uses a dataset of labeled text data and aims to find the best combination of weights for multiple classifiers to improve classification accuracy.

## Project Structure

- **`genetic_algorithm.py`**: Contains the main implementation of the genetic algorithm, including functions for data preparation, classifier training, initialization of parents, crossover, and fitness evaluation.
- **`parent_class.py`**: Defines the `Parent` class, representing an individual solution with weighted classifiers. Each `Parent` instance includes methods for evaluating fitness based on classifier predictions.

## Requirements

- Python 3.6 or later
- Dependencies: Install required libraries with:

  ```bash
  pip install pandas numpy scikit-learn
