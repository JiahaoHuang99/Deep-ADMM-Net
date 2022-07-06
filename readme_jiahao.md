## 

- data
    - Brain_data (CC dataset)
        - db_train_mat (CC Training set GT)
        - db_valid_mat (CC validation set GT)
    - Brain_data_sampling (Using `Gen_traindata.m` to generate ZF data here)
    - result (Result temp)
    - result_G1D30_CC (CC G1D30 Result Archive)
    - result_G2D30_CC (CC G1D30 Result Archive)

## Train

1. Use `Gen_traindata.m` to generate training data
    - Two setting
        - CC
        - fastMRI
2. Config Setting in `config.m`
2. Use `main_netTrain.m` to train the network

## Test



## Analysis


Deep ADMM Net is very sensitive to the undersampling mask. Train and test must use the same mask.