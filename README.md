# Face Mask Detection, Classification, and Segmentation

## Project Summary

This project develops a computer vision solution to detect, classify, and segment face masks in images. It combines traditional machine learning methods using handcrafted features with deep learning approaches. The work is divided into four main tasks:

- **Task a:** Binary classification using handcrafted features (HOG) with ML classifiers (SVM and shallow Neural Network).  
  - The Neural Network achieved **93.25%** validation accuracy.  
    - **Confusion Matrix (NN):**  
      - Actual with_mask: **332 correct**, **30 misclassified**  
      - Actual without_mask: **25 misclassified**, **428 correct**  
  - The SVM achieved approximately **91.66%** accuracy.

- **Task b:** Binary classification using a Convolutional Neural Network (CNN) with hyperparameter tuning.  
  - A grid search was performed over different learning rates, batch sizes, and optimizers (e.g., Adam and SGD) to select the best model configuration.

- **Task c:** Region segmentation using traditional techniques.  
  - Global Otsu’s thresholding combined with morphological operations was applied to extract mask regions from images.  
  - Although this method provides a reasonable baseline, it is less precise than deep learning methods.

- **Task d:** Mask segmentation using a U-Net model.  
  - A U-Net architecture with an encoder–decoder structure and skip connections was implemented for precise segmentation, evaluated using metrics such as IoU and Dice score.

## Directory Structure

```
root/
├── haarcascade_frontalface_default.xml    # Haar Cascade file for face detection
├── dataset/                               # Folder containing images in 'with_mask' and 'without_mask' subdirectories
├── MSFD/1/face_crop/                       # Training dataset for the U-Net model
├── MSFD/1/face_crop_segmentation/          # Testing dataset for the U-Net model 
├── task_a_b.ipynb                         # Notebook for Task a (handcrafted features & ML classifiers) and Task b (CNN-based classification)
└── task_c_d.ipynb                         # Notebook for Task c (traditional segmentation) and Task d (U-Net based segmentation)
```

## How to Run the Project

### Prerequisites

Ensure you have the following installed:
- **Python 3.x**
- **TensorFlow (>=2.x)**
- **scikit-learn**
- **OpenCV**
- **Matplotlib**
- **NumPy**

### Installation

You can install the required packages using pip:

```bash
pip install tensorflow scikit-learn opencv-python matplotlib numpy
```

### Running the Notebooks

1. **Task a & Task b:**
   - Open `task_a_b.ipynb` in Jupyter Notebook.
   - Run the cells sequentially to perform feature extraction, train the SVM and shallow Neural Network classifiers, and train the CNN with hyperparameter tuning.
   - Review the classification results, confusion matrices, and training curves.

2. **Task c & Task d:**
   - Open `task_c_d.ipynb` in Jupyter Notebook.
   - Run the cells sequentially to perform traditional region segmentation (using Otsu’s thresholding and morphological operations) and to train and evaluate the U-Net model for mask segmentation.
   - Review the segmentation outputs and evaluation metrics (e.g., IoU, Dice score).

## Project Dataset

To download the datasets used, visit the [Github Repo for Task-a & Task-b](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)  and  [Github Repo for Task-d](https://github.com/sadjadrz/MFSD).

---

