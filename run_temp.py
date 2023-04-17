from dataloader import Dataloader
from helper_functions import *
from lstm_models import *
from preprocess import *
from templates import *
import warnings
warnings.filterwarnings("default", category=DeprecationWarning)


train_X, train_y = get_ts_format(X_files, y_files, WINDOW_SIZE, STEP_SIZE)
print(train_X.shape, train_y.shape)