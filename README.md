# Project Title

PREDICTING THE BEHAVIOUR OF A DRIVER BY USING TELEMATICS AND MACHINE LEARNING TECHNIQUES.

## Getting Started

Please download the copy of the project source code and place into a folder in your system to execute the same. 

### Prerequisites

Softwares that are required to execute the machine learning model and python flask web application

``` 
Python - 3.6.4 Version 
IDE - PyCharm
```

### Installing

Follow the below steps one by one to run the application

```
Open the source code with the PyCharm IDE
```

Next Activate the Python virtual environment which you have configured

```
cd env/Scripts activate
```

Next install the required supporting packages using the Pip command

```
pip3 install -r requirements.txt 
```

Download the training dataset from the below mentioned URL and place the files inside the folder 'MadhaviBoyapati_2942211_Project/code/DriverBehaviourWithTelemetrics/features'

```
https://drive.google.com/drive/folders/1Z9-wGuBZL8j0eEaVc6KiVaCmVHFUqP53?usp=sharing
```

Next run the python flask application

```
python app.py 
```

Then open the URL in the brower which is running as a web instance. 

## Pre-Processing the Dataset for the training dataset

```
python preprocessing.py train  
```

## Pre-Processing the Dataset for the evaluation evaluate

```
python preprocessing.py evaluate  
```

## Train the model 

```
python train.py 
```

## Test the model with the input data  

```
Choose one of the sample test with telematics data from the folder - MadhaviBoyapati_2942211_Project/code/SampleInputDataForTests 
In the front-end user interface, we have an option to upload the input csv file and result will be displayed which includes the model prediction. 
```


## Authors

***Madhavi Boyapati** - 2942211 - Student of Griffith College Dublin. - Masters in Big Data Management and Analytics. 

## License

This project is open to use it for learning and make enhancements to the code without any restrictions. 

## Acknowledgments

I owe my sincere gratitude and thanks to my professor and guide Dr.Viacheslav Filonenko for guiding me throughout the process of implementing and documenting.
