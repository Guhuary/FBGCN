# FBGCN

## Requirements

  * Python 3.6.2
  * For the other packages, please refer to the requirements.txt.

#### Test/Train on benchmark datasets

1. If you have cloned this repository and downloaded our pre-trained models. You can directly run `./src/test.py` to test the model. It should be noted that the number of layers is controlled by  `--nhiddenlayer`, layers = 2 + nhiddenlayer in our model, please set `--nbaseblocklayer` to 1.

2. If you want to train a model by yourself, you can set parameters and run `./src/train.py`. The models will be saved in `./src/Models/`