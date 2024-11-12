# Project Title: 
### Semantic Segmentation task: Automated Sea Turtle Segmentation
The github URL: https://github.com/sdoqb15/Semantic-Segmentation-task-Automated-Sea-Turtle-Segmentation.git
# Group name:
### HHHD

-----------------------------------------------------------------------------------------------------------------------------------------------

All of our code is integrated within a Jupyter Notebook, which is organized into clear sections as follows:
#### **1. Preliminary Steps**
##### 1.1 Install necessary packges
##### 1.2 Import necessary liberaries
##### 1.3 Set the device to GPU
#### **2. Process the dataset**
##### 2.1 Download the dataset
##### 2.2 Observe the dataset
##### 2.3 Split the dataset (using 'split_open')
##### 2.4 Data Preprocessing and Mask, Dataset Creation
#### **3. Define Models**
##### 3.1 Mean-shift
##### 3.2 K-means
##### 3.3 U-Net
##### 3.4 DeepLabV3
##### 3.5 DeepLabV3Plus
##### 3.6 DeepLabV3Plus_MobileNetV2
##### 3.7 DeepLabV3Plus_VIT
##### 3.8 DeepLabV3Plus_VIT2
##### 3.9 DeepLabV3Plus_VIT3
#### **4. Train Models**
##### 4.1 Mean-shift
##### 4.2 K-means
##### 4.3 U-Net
##### 4.4 DeepLabV3
##### 4.5 DeepLabV3Plus
##### 4.6 DeepLabV3Plus_MobileNetV2
##### 4.7 DeepLabV3Plus_VIT
##### 4.8 DeepLabV3Plus_VIT2
##### 4.9 DeepLabV3Plus_VIT3
#### **5. Test models and show results**
##### 5.1 Mean-shift
##### 5.2 K-means
##### 5.3 U-Net
##### 5.4 DeepLabV3
##### 5.5 DeepLabV3Plus
##### 5.6 DeepLabV3Plus_MobileNetV2
##### 5.7 DeepLabV3Plus_VIT
##### 5.8 DeepLabV3Plus_VIT2
##### 5.9 DeepLabV3Plus_VIT3
##### 5.10 Summary of mIoU Results in test set of all the models
##### 5.11 Test the segmentation effect of specific images on the test set (compare all the models)

-----------------------------------------------------------------------------------------------------------------------------------------------

The notebook contains the outputs of all cells，you can directly review our work and results by looking at the outputs of each cell in the notebook. If you want to test our code, here are your options:

1. **Run the Notebook from Start to Finish**  
   The simplest way is to execute the entire Notebook from the beginning. However, be aware that this process will take a significant amount of time, primarily due to the lengthy training phase. Since we have 9 models, the complete execution will require more than 24 hours.

2. **Fastest Way**  
   To save time, you can download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1SW2LKlv_PPA5Lo6U--W6G__ymlBx9OnM?usp=sharing). Place the downloaded models in the same directory as this Notebook. Then, skip the **(4. Train models)** step and proceed directly to **(5. Test models and show results)**.

3. **Testing Training Code Efficiently**  
   If you wish to test our training code but also want to save time, you can choose to train only one model while downloading the other pre-trained models from the above link.

# 1. Preliminary Steps
All cells in this section **must be executed** before running the rest of the notebook.

To ensure that our results can be accurately reproduced, please make sure that the hardware and environment setup you're using are as close as possible to ours, especially if you plan to test the training steps.

### Our Main Hardware Information:
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU

### Our Main Environment Information:
- **Python Version**: 3.9.20
- **PyTorch**: 2.5.1 (with CUDA 12.4)
- **TorchVision**: 0.20.1
- **Torchaudio**: 2.5.1
- **TensorFlow**: 2.18.0
- **Timm**: 1.0.11
- **Scikit-Learn**: 1.5.2
- **Scipy**: 1.14.1
- **Pandas**: 2.2.3
- **Numpy**: 1.26.4
- **Transformers**: 4.46.1
- **SpaCy**: 3.8.2
- **Tokenizers**: 0.20.2
- **Gensim**: 4.3.2
- **Huggingface Datasets**: 2.20.0
- **PyArrow**: 16.1.0
- **Smart-Open**: 7.0.4
- **OpenCV**: 4.10.0.84

For more detailed environment information, please refer to the `environment.yml` file available on our [Google Drive](https://drive.google.com/drive/folders/1SW2LKlv_PPA5Lo6U--W6G__ymlBx9OnM?usp=sharing).

**Note**: You can run our code on Google Colab, but we cannot guarantee that the results will perfectly match due to differences in hardware and software environments.

# 2. Process the dataset
All cells in this section **must be executed** to properly process the dataset.

### About the Dataset
You don't need to manually prepare or download the dataset beforehand. We use the command `kagglehub.dataset_download("wildlifedatasets/seaturtleid2022")` to automatically download the dataset and to ensure my code can access it.
Our code will handle setting the correct path. 

**Note**: If you prefer to use a pre-prepared dataset, ensure that you modify the paths in the code accordingly.

# 3. Define Models
It is **recommended** to run all the cells in this section to prepare for loading or training the models in subsequent steps. However, you can choose to run only the cells for the specific model you wish to test or train.

**Note**: Even if you plan to load our pre-trained `.pth` model files, you still need to run this section to properly define the models.

Each method and model is explained in detail within the `""" """` comments under their respective functions or classes, including a description of their structure and reference sources.

# 4. Train Models
This section is **time-consuming**, and you may choose to skip it and directly use our pre-trained models. You can download the trained models from [Google Drive](https://drive.google.com/drive/folders/1SW2LKlv_PPA5Lo6U--W6G__ymlBx9OnM?usp=sharing) and place them in the same directory as this notebook. Afterward, you can proceed to the next step, **“5. Test models and show results,”** for testing.

If you would like to run the training process, you can select a specific model (e.g., `DeepLabV3Plus_VIT3`) and execute only the relevant cells to save time.

Our training code outputs a **training log** (displaying progress) and automatically saves the model with the **lowest validation loss** as a `.pth` file in the same directory as this notebook. Additionally, we save the training process as a `.json` file (including loss and IoU metrics over time) and use it to plot the training curves.

For the **Mean-shift** and **K-means** traditional models, there is no actual “training” involved. Instead, we manually adjust hyperparameters using the training set and apply them to the test set in the next section, ensuring no data leakage. These models do not require saving to disk.

If the training time is too long, you can reduce the number of `epochs` in **Section 2.4** from 30 to a smaller value (though we recommend at least 15 epochs to obtain meaningful results).

# 5. Test models and show results
Congratulations on Reaching the Most Exciting Part! By now, you've hopefully completed the necessary steps following the instructions provided earlier. You should have either trained your models or downloaded the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1SW2LKlv_PPA5Lo6U--W6G__ymlBx9OnM?usp=sharing). Regardless of the path you've chosen, you should have at least one `.pth` model file in your root directory. Now, let's proceed to run the cells associated with your model and conduct some testing! 

*(Note: Mean-shift and K-means, our traditional methods, can be tested directly without loading any models.)*

For each model, there are two main testing cells:
1. **Testing mIoU on the Entire Test Set**: This part is time-consuming, and our code will print a log message after processing every 50 images.
2. **Segmenting a Specific Image**: Here, you'll select an image and use the model to observe its segmentation results, complete with the output image and corresponding IoU values.

In **Section 5.10**, we summarize the mean IoU (mIoU) results for each class across all models in a comprehensive table.

If all 7 deep learning `.pth` model files are already present in your root directory (the same directory as this Notebook)—you can download all of them at once from the [Google Drive link](https://drive.google.com/drive/folders/1SW2LKlv_PPA5Lo6U--W6G__ymlBx9OnM?usp=sharing)—then you can execute all the cells in **Section 5.11** to simultaneously view the segmentation performance of all models on a specific image.
