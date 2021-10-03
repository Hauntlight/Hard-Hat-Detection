
# Traffic Light And Signal Detection
In this project, we present a supervised method of **detecting hard-hat** completely based on deep Convolutional Neural Networks (CNNs). We will use [YOLOv5](https://github.com/ultralytics/yolov5) (You Only Look Once) which uses [PyTorch](https://pytorch.org/), an open source neural network framework. Instead, to write and execute Python code we used **Google Colaboratory**, a free environment that runs entirely in the cloud and provides a GPU.

## How to train and how to improve object detection
After loading the project **folder** on the drive, we have created a new Google Colaboratory session for training the neural network.
* in Colaboratory file, need to change the runtime type: from *Runtime* menu select *Change runtime* type and choose **GPU** as Hardware accelerator.

## `Step 1. Configuration`
In this section we will deal with mounting Google Drive and installing the requirements necessary for the use of YOLOv5

We will proceed to mount Google Drive on the Colab session

```
from google.colab import drive  
print('Collego GDrive')  
drive.mount('/content/gdrive')  
%cd /content/gdrive/MyDrive/$root_folder/
```

Now we will proceed to clone the **[repository]( https://github.com/ultralytics/yolov5)** of YOLOv5 and install all requirements.Finally we carry out the import of the torch library.
```
!git clone https://github.com/ultralytics/yolov5  
!pip install -r yolov5/requirements.txt  
import torch  
print("Setup complete.")
```
Unfortunately it will be necessary to restart the runtime and rerun the cells in order to re-enter the entered parameters and relocate to the correct folder.

The folder provided already contains the dataset (a modified version of **[this]( https://public.roboflow.com/object-detection/hard-hat-workers/)**), so we are ready for training .


## `Step 2. Training`

Now we have 2 main possibilities: Start the training from scratch (what we actually performed for this project) or resume a training that was previously interrupted. We also have the possibility to choose the number of iterations to carry out and the type of pre-trained weight to use.
```
epoch_number = 300 #@param {type:"slider", min:100, max:10000, step:20}
    option = 'START TRAINING FROM BEGINNING' #@param ["START TRAINING FROM BEGINNING", "RESUME TRAINING"]
    weight = 'yolov5s.pt' #@param["yolov5s.pt","yolov5m.pt","yolov5l.pt","yolov5x.pt"]
    batch = 64 #@param {type:"slider", min:1, max:64, step:1}
    #@markdown Choosing the resume option ignore the epoch option and the weight option will be ignored

    if option == 'START TRAINING FROM BEGINNING':

        %rm -r /content/gdrive/MyDrive/$root_folder/risultato/exp
        !python /content/gdrive/MyDrive/$root_folder/yolov5/train.py --img 520 --batch $batch --epochs $epoch_number --data /content/gdrive/MyDrive/$root_folder/dataset/data.yaml --weights $weight --cache --project /content/gdrive/MyDrive/$root_folder/risultato/
        %rm -r /content/gdrive/MyDrive/$root_folder/lastTrain/exp
        %cp -r /content/gdrive/MyDrive/$root_folder/risultato/exp /content/gdrive/MyDrive/$root_folder/lastTrain/
    else :
        !python /content/gdrive/MyDrive/$root_folder/yolov5/train.py --resume /content/gdrive/MyDrive/$root_folder/risultato/exp/weights/last.pt 
        %rm -r /content/gdrive/MyDrive/$root_folder/lastTrain/exp
        %cp -r /content/gdrive/MyDrive/$root_folder/risultato/exp /content/gdrive/MyDrive/$root_folder/lastTrain/
```
 
We now describe what we have coded: when you decide to start a new training, the last training started is removed, and if this just launched should be successful, the old weight is automatically removed and replaced with the best one, obtained during training.

The speech is similar when you choose to carry out the resume from the last interrupted training.


## `Step 3. Detection`


When the training is complete, we will perform object detection on the videos and save the results on the Drive.

Run the following command lines, obviously by entering the name of the file on which to perform the object detection, which must be placed in the 'videos' folder:
```
    video_name = 'construction.mp4' #@param {type:"string"}

    !python /content/gdrive/MyDrive/$root_folder/yolov5/detect.py --source /content/gdrive/MyDrive/$root_folder/videos/$video_name --weights /content/gdrive/MyDrive/$root_folder/lastTrain/exp/weights/best.pt --project /content/gdrive/MyDrive/$root_folder/detection/
```
