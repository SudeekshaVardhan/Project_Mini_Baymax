# Project Mini Baymax

Modeled after Disney's lovable healthcare assitant robot Baymax, this program uses image recognition software to identify physical symptoms of diseases such as chickenpox and measles. Because these diseases are so similar in physical appearence, it may be difficult for patients to correctly identify what disease they have contracted, and what the necessary treatments are for that disease. As a result, due to the large number of training images, the program will be able to make out the differences between each diseases physical symptoms

## The Algorithm

Access the full dataset here: https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20?select=Original+Images

This program utilizes a pretrained model built on an existing dataset comprised of 6 different image classes from the fold 1 file of the original images directory on the dataset:
1. chickenpox
2. cowpox
3. healthy
4. hfmd
5. measles
6. monkeypox

The model uses a resnet18 network to process images, and was built using the Jetson Inference Library

## Steps to Build the Model:

*Note: "filename" is used in the commands to denote where the dataset name should have gone. The name used by me for the dataset was "fold1."

1. Download the file at the link below, navigate to the FOLDS directory (found in Original Images directory), and then select one of the folds to use for your dataset (fold 1 was used to train the existing model).

2. Create a labels.txt file for the dataset, and input each of the class names (one per line), compress the folder into a zip file

3. Remote-ssh into VSCode and log into the Jetson Nano (use the device ip address to initllly connect to the ssh host, and when prompted for password, use "nvidia") 

4. Drag and drop the compressed folder to VS Code under the directory jetson-inference/python/training/classification/data, and then unzip the zipped file using the 'unzip' command

5. Once the file is unzipped, cd back to the jetson-inference library, and then run the command below to ensure that there is enough memory to train the device:
        echo 1 | sudo tee /proc/sys/vm/overcommit_memory

6. Enter the docker container from the jetson-inference library: ./docker/run.sh

7. cd to the jetson-inference/python/training/classification file in the docker, and then run the command below to start training the model. (Note: Run the command without the batch-size command first, then run with the batch-size command if there is a memory allocation error in the terminal. This will reduce the accuracy of the data, however.):
        python3 train.py --epochs=10 --batch-size=2 --model=models/filename data/filename

8. Once the model has finished training, export the model with the code below:
        python3 onnx_export.py --model-dir=models/filename

9. Once the model is exported, exit the docker container. cd into the jetson-inference/python/training/classification/models/fold1, and then ensure that there is a resnet18.onnx file saved under the model name

10. The model is complete, and now it is ready to be trained.

## Running this project

*Note: image_file is used to denote the test image that is downloaded to the folder that is used in the my-recognition.py file to identify the image.

1. Open the my-recognition file in the GitHub repository in VS Code, then open a new terminal

2. Download an image online (image file is given), and upload it to the nano using the wget command, or by dragging and dropping an existing image on the PC to the project folder. Make sure that the directory remains in the same project folder as the my-recognition file.
        
3. From the new terminal, enter the command:
        python3 my-recognition.py image_file.jpg
