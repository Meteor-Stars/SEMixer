# SEMixer
Our experimental environments involve Pytorch 1.12.1 and Numpy 1.22.4.

## Datasets
  You can download the public datasets used in our paper from https://drive.google.com/drive/folders/1PPLsAoDbv4WcoXDp-mm4LFxoKwewnKxX. The downloaded folders e.g., "ETTh1.csv",  should be placed at the "dataset" folder. These datasets are extensively used for evaluating performance of various time series forecasting methods.
  
## Reproducing Scripts
The two cores of SEMixer are MPMC and the DI module. The ablation study have demonstrated the effectiveness of them, which can be reproduced in our code scripts.

Specifically, we have provided the running scripts (including hyperparameters) for SEMixer and baseline models (CI-TSmixer, FiLM, DLinear, PatchTST, TimeMixer, Scaleformer, Pathformer, and FiLM) on the public datasets. For instance, running the script "Run_SEMixer_TSF (Ours).py" can reprpduce the paper results of our SEMixer.
  
