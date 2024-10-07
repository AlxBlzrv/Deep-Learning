# Fake News Detection with LSTM

This project focuses on detecting fake news using machine learning techniques, specifically a Bidirectional LSTM neural network. The dataset used for this project is the **Fake News** dataset from Kaggle. The project includes data preprocessing, feature extraction, model building, and evaluation.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Results](#results)
- [License](#license)

## Dataset

The dataset can be found at: [Kaggle - Fake News Dataset](https://www.kaggle.com/datasets/algord/fake-news)

It contains the following columns:

- **title**: The title of the news article.
- **news_url**: URL of the article.
- **source domain**: The domain where the article was posted.
- **tweet_num**: The number of retweets.
- **real**: Label of the article where `1` indicates real news and `0` indicates fake news.

## Installation

1. Clone the repository or download the code.
2. Install the required libraries using the following command:

   ```bash
   pip install pandas numpy matplotlib seaborn nltk tensorflow
   ```

3. Ensure that the dataset is downloaded from Kaggle and stored in the correct directory (`/kaggle/input/fake-news/FakeNewsNet.csv`).

## Project Structure

```text
├── fake_news_detection_lstm.ipynb   # Jupyter Notebook with the code
├── README.md                        # This file
└── FakeNewsNet.csv                  # Dataset (to be downloaded)
```

## Preprocessing

- **Stopwords Removal and Lemmatization**: I first clean the text by removing stopwords using NLTK's `stopwords` and lemmatizing the words with `WordNetLemmatizer`.
  
- **Data Augmentation**: I shuffle the words in the title of each article to create new training examples, helping the model generalize better.

- **Tokenization**: The text is tokenized, and each sequence is padded to ensure uniform input size for the neural network.

## Model

I use a **Bidirectional LSTM** model to detect fake news. The model architecture includes:

- An embedding layer that maps each word to a dense vector.
- Two bidirectional LSTM layers for sequential learning.
- Dropout layers to prevent overfitting.
- A dense output layer with a sigmoid activation for binary classification (real or fake news).

### Model Summary:

- **Embedding Layer**: Input dimension of 5000 and output dimension of 128.
- **LSTM Layer 1**: Bidirectional LSTM with 64 units and L2 regularization.
- **LSTM Layer 2**: Bidirectional LSTM with 32 units and L2 regularization.
- **Dropout Layers**: Dropout rate of 0.5 to reduce overfitting.
- **Dense Output Layer**: A single neuron with sigmoid activation to classify the news as real or fake.

### Training

- The model is trained for 20 epochs with a batch size of 128, using binary cross-entropy as the loss function and Adam as the optimizer.

## Results

After training the model for 20 epochs, it achieved an accuracy of **90%** on the test set.

```bash
Test accuracy: 0.90
```

## License

This project is licensed under the MIT License.
