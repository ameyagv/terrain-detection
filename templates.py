# List of training, validation, and test X_files
X_files = ['TrainingData/subject_001_01__x.csv', 'TrainingData/subject_001_02__x.csv',
           'TrainingData/subject_001_03__x.csv', 'TrainingData/subject_001_04__x.csv',
           'TrainingData/subject_001_05__x.csv', 'TrainingData/subject_001_06__x.csv',
           'TrainingData/subject_001_07__x.csv', 'TrainingData/subject_002_02__x.csv',
           'TrainingData/subject_002_03__x.csv', 'TrainingData/subject_002_04__x.csv',
           'TrainingData/subject_002_05__x.csv', 'TrainingData/subject_003_01__x.csv',
           'TrainingData/subject_003_02__x.csv', 'TrainingData/subject_003_03__x.csv',
           'TrainingData/subject_004_01__x.csv', 'TrainingData/subject_004_02__x.csv',
           'TrainingData/subject_005_01__x.csv', 'TrainingData/subject_005_02__x.csv',
           'TrainingData/subject_005_03__x.csv', 'TrainingData/subject_006_01__x.csv',
           'TrainingData/subject_006_02__x.csv', 'TrainingData/subject_007_02__x.csv',
           'TrainingData/subject_007_03__x.csv', 'TrainingData/subject_007_04__x.csv',
           'TrainingData/subject_008_01__x.csv']

val_X_files = ['TrainingData/subject_002_01__x.csv', 'TrainingData/subject_001_08__x.csv']
test_X_files = ['TrainingData/subject_006_03__x.csv', 'TrainingData/subject_007_01__x.csv']

# List of training, validation, and test y_files
y_files = ['TrainingData/subject_001_01__y.csv', 'TrainingData/subject_001_02__y.csv',
           'TrainingData/subject_001_03__y.csv', 'TrainingData/subject_001_04__y.csv',
           'TrainingData/subject_001_05__y.csv', 'TrainingData/subject_001_06__y.csv',
           'TrainingData/subject_001_07__y.csv', 'TrainingData/subject_002_02__y.csv',
           'TrainingData/subject_002_03__y.csv', 'TrainingData/subject_002_04__y.csv',
           'TrainingData/subject_002_05__y.csv', 'TrainingData/subject_003_01__y.csv',
           'TrainingData/subject_003_02__y.csv', 'TrainingData/subject_003_03__y.csv',
           'TrainingData/subject_004_01__y.csv', 'TrainingData/subject_004_02__y.csv',
           'TrainingData/subject_005_01__y.csv', 'TrainingData/subject_005_02__y.csv',
           'TrainingData/subject_005_03__y.csv', 'TrainingData/subject_006_01__y.csv',
           'TrainingData/subject_006_02__y.csv', 'TrainingData/subject_007_02__y.csv',
           'TrainingData/subject_007_03__y.csv', 'TrainingData/subject_007_04__y.csv',
           'TrainingData/subject_008_01__y.csv']

val_y_files = ['TrainingData/subject_002_01__y.csv', 'TrainingData/subject_001_08__y.csv']
test_y_files = ['TrainingData/subject_006_03__y.csv', 'TrainingData/subject_007_01__y.csv']

WINDOW_SIZE = 30
STEP_SIZE = 1

X_HEADER = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
X_TIME_HEADER = ['time']
Y_HEADER = ['label']
Y_TIME_HEADER = ['time']