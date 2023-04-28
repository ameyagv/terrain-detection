## CSC591/ECE 542 - Neural Networks: Proj-C

### Team 116

- Ameya Vaichalkar (agvaicha)
- Keerthana Telaprolu (ktelapr)
- Vikram Pande (vspande)

This readme contains the directory structure. The code has modular structure and there are files for different functionalities. Final predictions are in the jupyter notebook file.

##### Following are the modules we built for the project:
- ```helper_functions.py```: contains the functions to plot the graphs of accuracy and loss against epochs, get the  evaluation metrics and function to assign weights to imbalanced classes.
- ```lstm_models.py```: functions of variants of LSTM models with different combinations of parameters such as LSTM, BiDirectional LSTM, Dropout, Learning Rate.
- ```notebook.ipynb```: contains the code for predictions with modular methods.
- ```preprocess.py```: contains methods to preprocess the data. Methods to normalize the data, encode the data, match frequency of X and Y, get windowed data to feed LSTM, get the data in time series format.
- ```templates.py```: contains templates of train and test datafiles and some constants 

#### How to run? Steps for running the code:
1. The directory contains all the necessary files, download/clone the repository.
2. Copy the data in the same  directory.
3. Run ```notebook.ipynb``` that generates the prediction files.

#### References:
[1] B. Zhong, R. L. d. Silva, M. Li, H. Huang and E. Lobaton, "Environmental Context Prediction for Lower Limb Prostheses With Uncertainty Quantification," in IEEE Transactions on Automation Science and Engineering, vol. 18, no. 2, pp. 458-470, April 2021, doi: 10.1109/TASE.2020.2993399.
