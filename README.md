# BERT-based Sentiment Analysis

## Overview
This repository provides a BERT-based sentiment classifier. The goal of this project is to enable sentiment analysis of app reviews, categorizing them as **positive**, **negative**, or **neutral** based on the text input.

## Repository Structure
```
BERT-based-Sentiment-Analysis/
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
├── dockerfile    # Docker image to build
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
 git clone https://github.com/Markz88/BERT-based-Sentiment-Analysis.git
 cd BERT-based-Sentiment-Analysis
```

### Install Dependencies
Create a virtual environment using **Python 3.11.4** (optional, but recommended), and install the dependencies:
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


## Docker Instructions

You can build and run this project using Docker for a fully isolated and reproducible environment.

### Step 1: Build the Docker Image

To build the Docker image, run the following command from the root of the repository (where the Dockerfile is located):

```bash
docker build -t name:tag .
```

- Replace `name` with your preferred image name.
- Replace `tag` with your desired tag (e.g., `latest`).

### Step 2: Train the Model

To train the model using the Docker container:

```bash
docker run -it name:tag python main.py --mode "train"
```

You can also enable GPU support (if available and your Docker setup allows it) by running:

```bash
docker run --gpus all -it name:tag python main.py --mode "train"
```

This will start training using the datasets provided in the `data/` directory and save the trained model accordingly.

### Step 3: Evaluate the Model

To evaluate the model on the test dataset:

```bash
docker run -it name:tag python main.py --mode "evaluate"
```

You can also enable GPU support (if available and your Docker setup allows it) by running:

```bash
docker run --gpus all -it name:tag python main.py --mode "evaluate"
```

This runs the evaluation mode and prints out performance metrics such as accuracy, precision, recall, and F1-score.

### Step 4: Predict Sentiment from a Review

To perform inference using the trained model:

```bash
docker run -it name:tag python predict.py --text "Your app review here"
```

You can also enable GPU support (if available and your Docker setup allows it) by running:

```bash
docker run --gpus all -it name:tag python main.py --text "Your app review here"
```

Replace `"Your app review here"` with the review you want to classify. The script will output the predicted sentiment.


## License
This project is open-source and available under the MIT License.

## References
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Google Play Scraper Package](https://github.com/JoMingyu/google-play-scraper)
- [Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)
