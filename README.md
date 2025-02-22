This is crop disease detection system which is developed with the help of Deep learning techniques supported by some XAI techniques.

**Required libraries**

**Python:** Flask, Numpy, Pandas, OpenCV, MatplotLib, bson, PIL.

**Machine Learning:** Tensorflow, keras, SKlearn, Scipy.

**XAI:** Lime, SHAP.

**Database:** MangoDB.

**Api:** OpenAI


**Steps to run the code:**

**Step 1: **

  Copy repository to Visual Studios code.

**Step 2: **

  Download the dataset from kaggle using the link and place it in crop disease detection directory. 
  
  Link - https://www.kaggle.com/datasets/kaushikkumar208/plant-disease-detection

**Step 3: **

  Download all the required libraries using the code below.
  
  Pip install Flask Numpy Pandas OpenCV MatplotLib bson PIL
  
  Pip install Tensorflow sklearn scipy
  
  pip install lime shap
  
  pip install Pymongo

**Step 4: **

  Run the training codes to train the disease detection model.
  
  Change the Train_dir and Val_dir with the training dataset location.
  
  Use codes below to run codes:
  
    python EfficientNetV2S_train.py
    
    python Plant_detection.py
    
    python Resnet152_train.py
    
  Save the trained models in a file named "Trained models" in crop disease detection folder.

**Step 5:**

  Run the application using the code below: 
  
    Python app.py

**Step 6:**
  Open the web application in Chrome using the web address below
  
    127.0.0.1:5000
