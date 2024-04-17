# Indian-Sign-Language-Recognition
Communication plays a vital role in human interaction, enabling knowledge dissemination and relationship building through various modes such as speech, facial expressions, and hand signs. However, individuals with hearing and speaking disabilities rely heavily on non-verbal forms of communication, particularly sign language. In India, where approximately five million people are deaf or hearing impaired, Indian Sign Language (ISL) serves as a primary mode of communication. Despite its importance, research in this area remains limited due to its complexity.

To address this gap, our project focuses on recognizing all ISL alphabets (A-Z) and digits (0-9) using the bag of visual words model and converting them into text/speech. Dual-mode recognition enhances accuracy, with machine learning classifiers such as KNN, SVM, logistic regression, and a convolutional neural network (CNN) employed for classification. The dataset, manually created with various hand orientations, follows an 80:20 train-test ratio.

The system seamlessly integrates text and speech output, utilizing the Pyttsx3 module to ensure uninterrupted speech playback during live video streaming. Additionally, a reverse recognition feature allows input via speech of English alphabet letters, mapped to corresponding signs and displayed from the database. Furthermore, linguistic diversity is promoted by forming words from detected signs and converting them into Kannada and Hindi using the Google API, thus expanding accessibility and inclusivity.

#### Research Paper Sign Language Detection For Deaf And Dumb People 

## Getting Started
### Pre-requisites
Before running this project, make sure you have following dependencies - 
* [pip](https://pypi.python.org/pypi/pip)
* [Python 3.7.1](https://www.python.org/downloads/)
* [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)
* [Opencv contrib](https://pypi.org/project/opencv-contrib-python/)
* [Anaconda Distribution](https://www.anaconda.com/products/individual)

### Dataset
Download the images from [here](https://drive.google.com/drive/folders/1SY67sDO2ROoOoBhTBIIDn17gStS0AvCB?usp=sharing)

Some images of the dataset are shown below:
<p align="center">
  <br>
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/dataset.png" width="800" height="750">   
 </p>

### Run Project
To run the project, perform the following steps -
1. ```git clone https://github.com/shag527/Indian-Sign-Language-Recognition.git```
2. ```conda create --name sign python=3.7.1```<br />
3. ```conda activate sign```. 
4. ```pip install -r requirements.txt```. 
5. ```cd to the GitHub Repo till Predict signs folder```. 

Command may look like: ```cd 'C:\Users\.....\Indian-Sign-Language-Recognition\Code\Predict signs\'```

6. ```python main.py```

A tkinter window like this will open.
<p align="center">
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/login.png" width="400" height="410">
  <br>
 </p>

7. Create your account to access the system.
8. Now, the main tkinter window will open.
<p align="center">
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/main.png" width="400" height="410">
   <br>
 </p>
9. Click on the desired button to access the respective service.

#### To create your own recognition system
1. To create your own dataset, following the steps given above, go to the create signs panel and create signs.
2. Now, divide the dataset into train and test by running the Dividing_Dataset.ipynb file in the preprocessing folder.
3. To create histograms and saving them to .csv file, run the create_train_hist.py and create_test_hist.py respectively by extrating the SURF features and clustering them using MiniKbatchMeans.
4. Lastly, go to the classification folder and run different python files to check the results. 
5. After saving the model, you can load the model for testing purposes.


## Workflow

<p align="center">
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/" width="400" height="410">
  <br>
 </p>

### Preprocessing
Here 2 methods for preprocessing are used. First one is the background subtraction using an additive method, in which the first 30 frames are considered as background and any new object in the frame is then filtered out. Second one uses the skin segmentation concept, which is based on the extraction of skin color pixels of the user.

<p align="center">
  <br>
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/mask.png">       <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/after mask.png">       <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/canny.png">
  <br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Mask &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;After applying mask &nbsp; &nbsp; &nbsp;&nbsp;Canny Edge detection
</p>
  <br>
  
### Feature Detction and Extraction:
The Speeded Up Robust Feature (SURF) technique is used to extract descriptors from the segmented hand gesture images. These descriptors are then clustered to form the similar clusters and then the histograms of visual words are generated, where each image is represented by the frequency of occurrence of all the clustered features. The total classes are 36.
<p align="center">
  <br>
  <img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/SURF.png">
 <br>
 &nbsp&nbsp&nbsp&nbsp&nbsp SURF Features
</p>

### Classification
In this phase, various classifiers are used in order to check the best classifier for prediction. The classifiers used are:

+ Naive Bayes
+ Logistic Regression 
+ K-Nearest Neighbours (KNN)
+ Support Vector Machine (SVM)
+ Convolution Neural Network (CNN)

#### Accuracy Plot
The accuracy rate of different classifiers obtained are shown below:
<p align="center">
  <br>
<img align="center" src="https://github.com/shag527/Indian-Sign-Language-Recognition/blob/master/Images/acc.png">
 </p>

### Output
The predicted labels are shown in the form of text as well as speech using the python text to speech conversion library, Pyttsx3.The system facilitates linguistic diversity by forming words from detected signs and converting them into Kannada and Hindi using the Google API, expanding accessibility and inclusivity.


### Reverse Sign Recognition
Dual mode of communication is implemented. The spoken word is taken as input and the corresponding sign images are shown in sequence. Google speech API is used for this purpose.

### Credits
+ [Bag of Visual Words (BOVW)](https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f)
+ [Image Classification with Convolutional Neural Networks](https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8)
