# A Graph-Embedded Latent Space Learning and Clustering Framework for Incomplete Multimodal Multiclass Alzheimer’s Disease Diagnosis

This repo contains the official Pytorch implementation of the paper: A Graph-Embedded Latent Space Learning and Clustering Framework for Incomplete Multimodal Multiclass Alzheimer’s Disease Diagnosis

## Contents

1. [Summary of the Model](#1-summary-of-the-model)
2. [Setup instructions and dependancies](#2-setup-instructions-and-dependancies)
3. [Running the model](#3-running-the-model)
4. [Some results of the paper](#4-some-results-of-the-paper)
5. [Contact](#5-contact)
6. [License](#6-license)

## 1. Summary of the Model

The following figure shows the overview for our proposed model Graph-SLC:

<img src= image\framework.png>

Our proposed Graph-SLC consists of three key components, including multimodal reconstruction module, subject-similarity graph embedding module, and AD-oriented latent clustering module. For a given subject, the multimodal reconstruction module first generates a subject-specific latent representation, which integrates information from different modalities under the guidance of all available modalities. Subsequently, the latent representation is constrained by the subject-similarity graph embedding module to maintain neighborhood relationships between the input subject and other subjects within the latent space. Finally, the latent representation is fed into the AD-oriented latent clustering module to enhance its separability between different disease categories.

## 2. Setup instructions and dependancies

For training/testing the model, you must first download ADNI dataset. You can download the dataset [here](https://adni.loni.usc.edu/data-samples/access-data/). Also for storing the results of the validation/testing datasets, checkpoints and loss logs, the directory structure must in the following way:

    ├── data                # Follow the way the dataset has been placed here
    │   ├── whole_Abeta       # Here Abeta-PET images must be placed
    │   ├── whole_FDG          # Here FDG-PET images must be placed
    │   ├── whole_Tau          # Here Tau-PET images must be placed
    │   └── whole_MRI          # Here MR images must be placed
    ├── data_info          # Follow the way the data info has been placed here
    │   ├── info.csv       # This file contains labels, age and gender information for each ID
    │   ├── whole.txt           # This file contains IDs of the dataset, like '037S6046'
    ├── result             # Follow the way the result has been placed here
    │   ├── exp_1              # for experiment 1
    │   │   └── CHECKPOINT_model.pth.tar     # This file is the trained checkpoint
    │   │   └── loss_curve.csv              # This file is the loss curve
    │   │   └── validation.csv              # This file is the indicator files in the validation set
    │   │   └── test.csv                    # This file is the indicator files in the test set
    ├── config.py          # This is the configuration file, containing some hyperparameters
    ├── dataset.py         # This is the dataset file used to preprocess and load data
    ├── main.py            # This is the main file used to train and test the proposed model
    ├── model.py           # This is the model file, containing two models (text_encoder and Unet)
    ├── README.md
    ├── utils.py           # This file stores the helper functions required for training

## 3. Running the model

Users can modify the setting in the config.py to specify the configurations for training/validation/testing. For training/validation/testing the our proposed model:

```
python main.py
```

## 4. Some results of the paper

Some of the results produced by our proposed model and competitive models are as follows. *For more such results, consider seeing the main paper*

<img src=image\result.png>

## 5. Contact

If you have found our research work helpful, please consider citing the original paper.

If you have any doubt regarding the codebase, you can open up an issue or mail at ouzx2022@shanghaitech.edu.cn

## 6. License

This repository is licensed under MIT license