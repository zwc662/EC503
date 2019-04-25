1. Run data_proc.py to preprocess the original dataset and build SMOTE as well as ADASYN datasets. The program should generate train.csv, train_SMOTE.csv and train_ADASYN.csv files in this folder.

2. Run run_nn.py to use SMOTE dataset to train an MLP for 100 epochs. Each epoch shall use out the whole SMOTE dataset. After training, nn will use the preprocessed dataset to test the trained model. The accuracy, precision and recall will be printed in the terminal.
