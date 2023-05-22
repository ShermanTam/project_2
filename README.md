# ImageCaptionGenerator
Image captioning trained using LSTM and Attention models along with audio generation
1. Click on the “Actions tab” in the Github repository
2. Click on the workflow name, ImageCaptionGenerators, available on the left pane
3. Once the workflow page opens, you will see a dropdown menu labeled "Run workflow."
Click on it
4. In the dropdown menu, set the respective values for the input parameters:
  a. **Model Type**: LSTM or Attention
  b. **Epoch Number**: 2, 5, 10, 20, 50, 70, 100
  c. ** Batch Size**: 5, 10, 14, 20, 32, 64, 128
6. The workflow will be triggered manually, and you can refresh the page to see the log of
the current workflow run. On the log page, you will find a "Build" tab. Click on the
"Build" tab to access the steps of the workflow
7. The log will provide a detailed description of each step being executed, starting from
downloading the dataset, preprocessing, training the model for the specified number of
epochs and batch size, and displaying the training results
8. Once the workflow job finishes, you will see the build logo in green color, indicating the
successful training of the model from the GitHub repository
