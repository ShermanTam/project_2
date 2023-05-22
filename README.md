# ImageCaptionGenerator
Image captioning is the task of generating human-like descriptions or captions for images. Our project aims to leverage the power of deep learning models to automatically analyze visual content and generate accurate and contextually relevant captions. We aim to use two models, VGG16 + LSTM and Inception V3 + Bahdanau Attention, for performance comparison. 
The codes (both in oops format and Google colab code which was developed initially) are provided in the repository with the names BaselineModel.py, AttentionModel.py, Baseline (not in oops), and Attention (not in oops). In order to try the training of our models from Github, Github actions have been utilized. The dataset files (zip) are uploaded in a drive folder that is set to public access, reducing the user's manual work by eliminating the requirement to download them and carry out additional tasks when training from Github. Here is the link for the dataset. (https://drive.google.com/drive/folders/1WNHl00Xuxh8-R2-VpJR5GLKszBkCOX83)
This repository contains code for training image captioning models using LSTM and Attention architectures, as well as audio generation capabilities. To train our models using the GitHub actions, follow these steps:

**Usage**
To use the image captioning models, follow these steps:

1.	Go to the GitHub repository and click on the "Actions" tab.
2.	Locate the workflow named "ImageCaptionGenerators" on the left pane.
3.	Open the workflow page and find the dropdown menu labeled "Run workflow." Click on it.
4.	Set the respective values for the input parameters in the dropdown menu:
    **Model Type**: Choose between LSTM or Attention.
    **Epoch Number**: Select the desired number of epochs (2, 5, 10, 20, 50, 70, or 100).
   ** Batch Size:** Choose the preferred batch size (5, 10, 14, 20, 32, 64, or 128).
    Note: The LSTM model uses VGG16 as the pretrained model, while the Attention model uses InceptionV3 with Bahdanau Attention.
5.	Manually trigger the workflow by clicking on the "Run workflow" button.
6.	Refresh the page to view the log of the current workflow run. You can find a "Build" tab on the log page, click on it to access the workflow steps.
7.	The log will provide a detailed description of each step being executed, including dataset downloading, preprocessing, model training, and training results.
8.	Once the workflow job finishes successfully, you will see the build logo in green color, indicating the completion of the model training process.

Note: For audio generation, please run the .py files separately in a different Integrated Development Environment (IDE) or execution environment of your choice.


