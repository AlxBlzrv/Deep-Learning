# X-ray Diagnostics

This project focuses on building and evaluating deep learning models to classify chest X-ray images into categories of normal and pneumonia. The main model architectures include a simple Convolutional Neural Network (CNN) and a fine-tuned pre-trained **VGG16** model using transfer learning.

## Project Overview

Chest X-ray images are used in medical diagnostics to detect pneumonia, a potentially life-threatening condition. Accurate classification is essential in healthcare, where false predictions could have serious consequences. This project explores two key model architectures:

- **CNN Model**: Custom-built convolutional neural network that achieves 87.66% accuracy.
- **VGG16 Model**: A pre-trained VGG16 network with fine-tuning, achieving a more reliable accuracy of 91,83%.

### Dataset

The project uses the following dataset for training, validation, and testing:

- **Kaggle Dataset**: [Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Key Features
- Image data preprocessing using augmentation techniques.
- Training custom CNN and fine-tuning a pre-trained VGG16 model.
- Evaluation of models based on accuracy and loss for both training and validation datasets.
- Visualization of the training process through graphs of accuracy and loss.

## Getting Started

To run the project, clone the repository and open the Jupyter Notebook included in the repository.

```bash
git clone https://github.com/AlxBlzrv/Deep-Learning.git
cd X-ray-Diagnostics
```

### Prerequisites

- Python 3.x
- TensorFlow/Keras
- Matplotlib
- Jupyter Notebook

To install the required dependencies:

```bash
pip install tensorflow matplotlib jupyter
```

## Acknowledgements

- **Data**: The dataset used in this project is sourced from Mendeley: [Chest X-ray Images (Pneumonia)](https://data.mendeley.com/datasets/rscbjbr9sj/2)  
- **License**: CC BY 4.0  
- **Citation**: For the original dataset, please cite the paper: [Cell Journal](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
- **Kaggle**: Additional dataset reference from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
