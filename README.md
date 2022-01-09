# Nottingham Histologic Grade Identification with Deep Neural Networks

## Introduction

Breast cancer grade identification is a labor-intensive task and needs human experts and time to continue the diagnosis.

This research aims to a new computer-aided diagnosis approach to continue the existing laboratory grading procedure with higher accuracy rates.

## Data Sources

[1] H. Bolhasani, E. Amjadi, M. Tabatabaeian, and S. J. Jassbi, “A histopathological image dataset for grading breast invasive ductal carcinomas,” Inform. Med. Unlocked, vol. 19, no. 100341, p. 100341, 2020.

[2] F. A. Spanhol, L. S. Oliveira, C. Petitjean, and L. Heutte, “A Dataset for Breast Cancer Histopathological Image Classification,” IEEE Trans. Biomed. Eng., vol. 63, no. 7, pp. 1455–1462, 2016.
## Methodology

Trained 3 DNN models with transfer learning (DenseNet) including, 2-predict, 3-predict, and
4-predict to classify the grades into benign-0, 1, 2, and 3.

#### Note:

weight20.h5: trained model for 2-predict (0-benign, 1-malignant)

weight30.h5: trained model for 3-predict (1-grade 1, 2-grade 2, 3-grade 3)

weight25.h5: trained model for 4-predict (0-benign, 1-grade 1, 2-grade 2, 3-grade 3)

### Implementation of the Machine Learning Models

Implemented the machine learning models  with
Spyder and Anaconda framework.

Libraries, dependencies used:  TensorFlow, Keras,
sklearn, and OpenCV.

Apart from that, a Flask Application Program
Interface with Anaconda framework was developed to
obtain the predictions.

### Implementation of the Inference Tool

Implemented an inference tool with Apache NetBeans with
Maven dependencies and used MongoDB as the
database.

Apart from that, integrated the Application Program Interface
to obtain the predictions.

## Results

Evaluated the models with the reserved test datasets and obtained more than 94% accuracy rates for all the trained models.

#### Find the research work at: @@@@@@@@@@@@@@@@


