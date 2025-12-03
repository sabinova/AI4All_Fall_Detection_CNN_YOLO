
# Computer Vision for Fall Detection and Alert in Elderly Care

Developed a YOLOv8-based computer vision system that detects falls in images of older adults, using custom-labeled Kaggle datasets to explore how AI can support elderly safety while revealing the impact of data imbalance, annotation choices, and bias in real-world healthcare scenarios through the AI4ALL Ignite program.

***

## Problem Statement

Falls are a leading cause of injury, hospitalization, and loss of independence among older adults, and many incidents go unnoticed when seniors live alone or are not continuously monitored.  This project investigates whether a deep learning–based computer vision model can automatically detect falls from camera images and provide timely alerts to caregivers, with the long-term goal of improving response time, supporting independent living, and highlighting ethical questions around privacy and fairness in AI for healthcare.

***

## Key Results

1. Trained a YOLOv8 object detection model on a custom fall dataset containing 1,784 images with 1,188 Fall and 2,076 Not_Fall annotations split into train, validation, and test sets.
2. Achieved strong detection performance on the validation set: overall precision ≈ 0.94, recall ≈ 0.92, mAP@50 ≈ 0.97, and mAP@50–95 ≈ 0.83, with slightly higher scores on the majority Not_Fall class.
3. Generated and analyzed multiple evaluation visualizations, including loss curves, precision–recall and F1–confidence curves, and normalized confusion matrices, to understand error patterns and class imbalance effects.
4. Produced qualitative demos on unseen validation images showing the model correctly localizing and classifying many realistic fall and non-fall scenarios involving diverse older adults in everyday environments.
5. Identified key sources of bias such as simulated falls, class imbalance, and manual annotation subjectivity, and outlined mitigation steps including dataset merging, rebalancing, and expanded real-world testing in future work.

***

## Methodologies

YOLOv8 (a supervised CNN-based object detector) was fine-tuned using transfer learning on Roboflow-annotated images, with bounding boxes labeled as **Fall** and **Not_Fall**.  Training was conducted in Google Colab over 30 epochs, using pre-trained YOLOv8 weights, standard augmentation, and separate train/validation/test splits to monitor generalization.  Evaluation combined quantitative metrics (precision, recall, mAP@50, mAP@50–95, and confusion matrices) with qualitative inspection of prediction samples to understand where the model succeeds or fails, especially under class imbalance and staged fall conditions.  The project also critically examined how dataset composition, manual labeling choices, and deployment settings can amplify or mitigate bias in AI systems used for elderly care.

***

## Data Sources

Primary datasets for this project:

- **Fall Detection – Eldercare Robot (Kaggle)**  
  https://www.kaggle.com/datasets/elwalyahmad/fall-detection

- **Fall Elderly People Detection (Kaggle)**  
  https://www.kaggle.com/datasets/ahmedh72/fall-elderly-people-detection

One dataset provided YOLO-style labels, while the other required manual annotation of 16k+ images using Roboflow to create bounding boxes and class labels before training.

***

## Technologies Used

- Python, Google Colab, Jupyter Notebooks
- YOLOv8 (Ultralytics) for object detection
- PyTorch (via Ultralytics backend)  
- Roboflow for image annotation and dataset export  
- NumPy, pandas, Matplotlib/Seaborn for data analysis and visualization
- Kaggle for dataset hosting and experimentation
- Git and GitHub for version control and collaboration

***

## Authors

This project was completed in collaboration with:

- **Sabina Ruzieva** – sabinaruzieva04@gmail.com  
- **Fiorella Rodriguez** – fiorellarodriguez120205@gmail.com  

Developed as part of the **AI4ALL Ignite** program, focusing on responsible AI, computer vision, and bias-aware machine learning for community impact.