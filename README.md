target-oriented attentive user/item embedding for recommendation
===
Datasets
---
  The dataset in this example is *delicious*, which has been processed and split into training (70%), validation(10%) and testing(20\%). The processed data is stored in data.pkl.
  <br>You need to place other supporting data files in the `/Data/` directory to the same directory with model.py to train and test the model.

Pre-Train
---
The pre-trained model is located in `/checkpoint/` directory, which can be used to initialized the model. 

Training and testing
---
run **`python3 model.py train`** to train the model and **`python3 model.py test`** to perform testing
<br>The negative sampling rate is different for the training, validating and testing phase, and it can be modified in the source file
<br>The recommendation performance is evaluating using HR@K AND NDCG@K, and they reside in the file `res.dat` in the directory `/model/` after testing
