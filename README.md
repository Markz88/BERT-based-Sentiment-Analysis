# BERT-based Sentiment Analysis

## Overview
This repository provides a BERT-based sentiment classifier. The goal of this project is to enable sentiment analysis of app reviews, categorizing them as **positive**, **negative**, or **neutral** based on the text input.

## Repository Structure
```
BERT-SentAnalysis/
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│
├── src/
│   ├── dataset.py  # Handles dataset loading and preprocessing
│   ├── model.py    # Defines the BERT-based model
│   ├── train.py    # Training functions
│   ├── evaluate.py # Evaluation functions
│
├── main.py       # Main script to train and evaluate the model
├── predict.py    # Script for making predictions using the trained model
├── requirements.txt # List of dependencies
```

## Dataset
The dataset used for fine-tuning the model consists of Google Play Store reviews collected using the Google Play Scraper for Python. The data has been preprocessed and split as follows:
- **Training Set (train.csv):** 80% of the data
- **Validation Set (val.csv):** 10% of the data
- **Test Set (test.csv):** 10% of the data

Each dataset entry contains:
- `content`: The review text from the Google Play Store
- `label`: The sentiment label (0: negative, 1: neutral, 2: positive)

## Installation
To run this project, you need to install the required dependencies.

### Clone the Repository
```bash
 git clone https://github.com/yourusername/BERT-Sentiment-Analysis.git
 cd BERT-Sentiment-Analysis
```

### Install Dependencies
Create a virtual environment (optional but recommended) and install dependencies:
```bash
pip install -r requirements.txt
```

## Training and Evaluating the Model
The `main.py` script is responsible for both training and evaluation.

### Training the Model
Run the following command to train the model:
```bash
python main.py --mode "train"
```
This script loads the dataset, fine-tunes the BERT model, and saves the trained model.

### Evaluating the Model
To evaluate the trained model on the test dataset:
```bash
python main.py --mode "evaluate"
```
This script computes performance metrics such as accuracy, precision, recall, and F1-score.

## Making Predictions
To use the trained model for sentiment prediction on new text:
```bash
python predict.py --text "Your app review here"
```

## License
This project is open-source and available under the MIT License.

## References
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Google Play Scraper Package](https://github.com/JoMingyu/google-play-scraper)
- [Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)
