# Music Recommendation System


## Directory and files organization
Directories
| Name  | Description |
| ------------- | ------------- |
| models  | all models  |
| configs | config files for experiments |
| results | folder to store inference results |
| save | trained model weight |


Files
| Name | Description |
| --- | ------------ |
| main.py | entry point for training |

Data path
`musicRatings.csv`

## Quickstart
```bash
python3 main.py --config_file configs/dense_ncf.yml
```

## Dataset
The dataset we are using is a private dataset collected by NYU Fox Lab. If you want access, please contact. 

## Model Performance
Regression task for the ratings
| Model name | MSE loss | config_file |
| :---: | :---: |     :---:     |
| NCF  | 1.324 |  configs/dense_ncf.yml |
| DeepFM | 6.379 | configs/dense_deepfm.yml |
