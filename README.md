# Skin Cancer Prediction Using EfficientNetB0 with HAM10000 Dataset

This repository contains the code and supporting files for a skin cancer prediction project using the EfficientNetB0 model. The project utilizes a modified version of the HAM10000 dataset, enhanced with additional images of healthy skin, to classify various types of skin cancer lesions. The model is trained using TensorFlow and provides detailed visualizations for predictions.

Check out my LinkedIn article or medium article to get detailed walkthrough article.
LinkedIn: [https://www.linkedin.com/in/yadavsidhant](https://www.linkedin.com/pulse/skin-cancer-prediction-images-using-cnn-based-based-ham10000-yadav-ezilc?trk=public_post_feed-article-content)
Medium: [https://medium.com/@sidhantyadav](https://medium.com/@sidhantyadav/skin-cancer-prediction-by-images-using-cnn-based-efficientnetb0-based-on-ham10000-dataset-8cab0ae185bb)

Or check out my Youtube Video for detailed walkthrough of the tutorial in video where I've explained all the functioning in detailed manner with the dataset modification.
Youtube: https://www.youtube.com/watch?v=xU_rFDf2c2E

Table of Contents
-----------------

-   [Introduction](#introduction)
-   [Dataset](#dataset)
-   [Model Architecture](#model-architecture)
-   [Training the Model](#training-the-model)
-   [Prediction](#prediction)
-   [Results Visualization](#results-visualization)
-   [Requirements](#requirements)
-   [Usage](#usage)
-   [File Descriptions](#file-descriptions)
-   [Acknowledgements](#acknowledgements)
-   [License](#license)

Introduction
------------

Skin cancer is one of the most common cancers worldwide. Early detection and accurate diagnosis are crucial for effective treatment. This project aims to leverage deep learning techniques to classify different types of skin cancer from dermatoscopic images. The EfficientNetB0 model is used due to its balance of efficiency and accuracy.

Dataset
-------

The dataset used in this project is a modified version of the HAM10000 dataset, which includes images of various skin lesions. In addition to the original dataset, we have included 109 images of normal, healthy skin to enhance the model's ability to distinguish between healthy and cancerous skin.

Model Architecture
------------------

The model is built using the EfficientNetB0 architecture, which is pre-trained on ImageNet and fine-tuned on our dataset. The architecture includes:

-   **EfficientNetB0 base model:** Used for feature extraction.
-   **Global Average Pooling layer:** Reduces each feature map to a single value.
-   **Dense layers:** Added on top for classification.

Training the Model
------------------

The model is trained with data augmentation techniques to improve its robustness. The training and validation data are generated using the `ImageDataGenerator` class from Keras with various augmentations such as horizontal and vertical flips, zoom, shear, and rotation.

Prediction
----------

The model can predict the type of skin cancer for a given image. The prediction script also provides detailed visualizations, including:

-   30x30 grid markings on the image.
-   Bounding boxes around detected lesions.
-   Confidence grid for the prediction.

Results Visualization
---------------------

The results of the predictions are visualized using Matplotlib, showing the original image, grid mask, bounding box, and a confidence grid.

Requirements
------------

To run this project, you need the following Python packages:

-   tensorflow
-   tensorflowjs
-   pandas
-   matplotlib
-   scikit-learn
-   opencv-python

Install the packages using the following command:

bash

`pip install tensorflow tensorflowjs pandas matplotlib scikit-learn opencv-python`

Usage
-----

1.  **Clone the repository:**

    bash

    `git clone https://github.com/yadavsidhant/skin-cancer-prediction.git`

2.  **Upload the dataset and CSV files:**

    -   Place the modified HAM10000 dataset and additional healthy skin images in the appropriate directories.
    -   Ensure the CSV files (`HAM10000_metadata.csv` and `ISIC2018_Task3_Test_GroundTruth.csv`) are in the root directory.
3.  **Run the training script:**

    python

    `python train.py`

4.  **Run the prediction script:**

    python

    `python predict.py --image_path path_to_image`
    
5. **Rin the Python Notebook (Recommended)**

   Open the python notebook 'Skin_Cancer_Prediction_HAM10000.ipynb' and mount the google drive for the dataset and can also be runned over google colaboratory.
   
File Descriptions
-----------------

-   `train.py`: Script to train the EfficientNetB0 model on the modified dataset.
-   `predict.py`: Script to predict skin cancer type for a given image and visualize the results.
-   `HAM10000_metadata.csv`: Metadata file for the HAM10000 training dataset.
-   `ISIC2018_Task3_Test_GroundTruth.csv`: Metadata file for the test dataset.
-   `class_indices.json`: JSON file containing class indices.
-   `skin_cancer_model.h5`: Trained model file.

Acknowledgements
----------------

-   **HAM10000 Dataset:** Provided by the Harvard Dataverse.
-   **EfficientNet:** Developed by Google Research, Brain Team.

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

* * * * *

We hope this project serves as a valuable resource for understanding and applying deep learning techniques to medical image analysis. Contributions are welcome!

* * * * *

For any questions or issues, please open an issue on the GitHub repository.
