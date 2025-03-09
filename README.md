# Lung Cancer Detection Using CNN

## Project Overview
This project implements a **deep learning-based lung cancer detection system** using **Convolutional Neural Networks (CNNs)**. The goal is to classify histopathological lung images into three categories:
- **Lung Adenocarcinoma (lung_aca)**
- **Lung Squamous Cell Carcinoma (lung_scc)**
- **Lung Benign tissue (lung_n)**

The model is trained on a **subset of the Lung and Colon Cancer Histopathological Images dataset from Kaggle** and achieves a test accuracy of **91.06%**.

## Dataset
The dataset consists of **1500 histopathological lung images** (500 images per class), extracted from the larger Kaggle dataset.
- The images are preprocessed and standardized before training.
- The dataset is split into **training (80%)**, **validation (10%)**, and **testing (10%)**.

## Model Architecture
A **custom CNN model** is implemented with the following structure:
- **Four convolutional layers** with ReLU activation and 3x3 kernels
- **MaxPooling layers** to reduce dimensionality
- **Fully connected layers** with dropout to prevent overfitting
- **Softmax activation** for multi-class classification

## Training & Evaluation
- The model is trained using **categorical cross-entropy loss** and the **Adam optimizer**.
- Training accuracy: **89.58%**
- Test accuracy: **91.06%**
- Evaluation metrics include accuracy, loss curves, and confusion matrices.

## Files in this Repository
- `Final_MV_Project.ipynb` - Jupyter Notebook containing the full implementation
- `MV_REPORT_FINAL.pdf` - Project report with detailed methodology and results

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Lung-Cancer-Detection-CNN.git
   cd Lung-Cancer-Detection-CNN
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Final_MV_Project.ipynb
   ```

## Future Improvements
- Implement **transfer learning** with pre-trained models like **ResNet or VGG16**.
- Increase dataset size for better generalization.
- Deploy as a web application for easy accessibility.

## Credits
- **Developed by:** Gauri Shashank Moghe
- **Dataset Source:** [Kaggle - Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

## License
This project is released under the **MIT License**.

