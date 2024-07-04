
# Abstract Gallery Generation using GANs

This project utilizes Generative Adversarial Networks (GANs) to generate abstract art images. The `gan-art.ipynb` notebook demonstrates the complete process of training a GAN to produce artistic images resembling those found in an abstract art gallery.

## Overview

Generative Adversarial Networks are a class of deep learning models used in unsupervised machine learning, where two neural networks, the generator and the discriminator, compete against each other. The generator learns to create realistic images, while the discriminator learns to distinguish between real and generated images. This dynamic competition leads to the generator producing increasingly realistic outputs.

## Project Structure

- **Abstract_Gallery_Generation/**:
  - **gan-art.ipynb**: Jupyter notebook containing the complete code for training and generating abstract art images using GANs.
  - **Abstract_gallery**: Input data or datasets used in training.
  - **README.md**: This file, providing an overview and instructions for the project.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm
- numpy

## Usage

1. **Clone the repository:**
   ```
   git clone https://github.com/AlxBlzrv/Deep-Learning.git
   cd Abstract_Gallery_Generation
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook:**
   ```
   jupyter notebook gan-art.ipynb
   ```

4. **Follow the instructions in the notebook to train the GAN and generate abstract art images.**


## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Feel free to modify and experiment with the code to create your own variations of abstract art using GANs. If you have any questions or feedback, please feel free to reach out.
