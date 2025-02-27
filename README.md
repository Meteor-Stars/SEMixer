# SEMixer
The experiment is conducted on pytorch.
## Datasets
  You can download the public datasets used in our paper from https://drive.google.com/drive/folders/1PPLsAoDbv4WcoXDp-mm4LFxoKwewnKxX. The downloaded folders e.g., "ETTh1.csv",  should be placed at the "dataset" folder. These datasets are extensively used for evaluating performance of various time series forecasting methods.
  
## Reproducing Scripts
### Input lengths for each method
In our experiments, we did not search for the hyperparameters of the model network. The key parameter for reproducing the results of the paper is the input length of the historical sequence. As mentioned in the paper, to evaluate the model's accuracy under both fixed and varying lengths, the input length for datasets Illness, Exchange and Solar energy is fixed at 512. For other datasets, the best input lengths searched for each method are shown in following Table (**also shown in Appendix Table 7**):

| Dataset |       | SEMixer | CI-TSMixer | DLinear | PatchTST | TimeMixer† | FiLM† | Scaleformer† | Pathformer† | Itransformer |
|---------|-------|---------|------------|---------|----------|------------|-------|--------------|-------------|--------------|
| **ETTh1** | 96    | 1280    | 1280       | 1280    | 640      | 1024       | 256   | 512          | 384         | 512          |
|          | 192   | 1280    | 1280       | 1280    | 512      | 384        | 512   | 512          | 512         | 640          |
|          | 336   | 384     | 1280       | 1280    | 1792     | 640        | 512   | 512          | 768         | 640          |
|          | 720   | 384     | 1280       | 1664    | 1792     | 384        | 512   | 512          | 384         | 384          |
| **ETTh2** | 96    | 1280    | 512        | 1536    | 512      | 512        | 128   | 640          | 384         | 512          |
|          | 192   | 1024    | 640        | 1664    | 512      | 384        | 512   | 384          | 384         | 640          |
|          | 336   | 1024    | 1280       | 512     | 1024     | 384        | 512   | 256          | 512         | 384          |
|          | 720   | 768     | 512        | 384     | 384      | 512        | 512   | 256          | 384         | 384          |
| **ETTm1** | 96    | 768     | 384        | 2048    | 384      | 384        | 256   | 384          | 512         | 384          |
|          | 192   | 1536    | 384        | 2048    | 1664     | 384        | 512   | 256          | 512         | 384          |
|          | 336   | 1536    | 384        | 2048    | 1664     | 384        | 1280  | 256          | 512         | 640          |
|          | 720   | 1664    | 1792       | 2048    | 1664     | 1280       | 1536  | 1024         | 512         | 2048         |
| **ETTm2** | 96    | 768     | 384        | 1664    | 768      | 640        | 128   | 256          | 384         | 512          |
|          | 192   | 1664    | 384        | 1664    | 1024     | 1792       | 128   | 256          | 384         | 2048         |
|          | 336   | 1664    | 384        | 2048    | 1024     | 384        | 256   | 256          | 768         | 1280         |
|          | 720   | 1664    | 2048       | 1664    | 1536     | 512        | 1536  | 384          | 768         | 1280         |
| **Weather** | 96   | 2048    | 512        | 2048    | 1280     | 384        | 128   | 512          | 384         | 384          |
|           | 192   | 2048    | 512        | 2048    | 1024     | 384        | 128   | 384          | 384         | 384          |
|           | 336   | 2048    | 384        | 2048    | 1024     | 384        | 384   | 640          | 512         | 384          |
|           | 720   | 2048    | 512        | 2048    | 1280     | 640        | 768   | 768          | 512         | 512          |
| **Electricity** | 96 | 1664    | 1664       | 2048    | 512      | 512        | 512   | 1664         | 384         | 1664         |
|              | 192 | 1664    | 1664       | 2048    | 512      | 512        | 512   | 1664         | 384         | 1664         |
|              | 336 | 1664    | 1664       | 2048    | 512      | 512        | 512   | 1664         | 384         | 1664         |
|              | 720 | 1664    | 1664       | 2048    | 512      | 512        | 512   | 1664         | 384         | 1664         |



### Reproducing
The two cores of SEMixer are MPMC and the DI module. The ablation study have demonstrated the effectiveness of them, which can be reproduced in our code scripts.

Specifically, we have provided the running scripts "Run_SEMixer_Baselines_TSF.py", which includes used hyperparameters for SEMixer and baseline models (CI-TSmixer, FiLM, DLinear, PatchTST, TimeMixer, FiLM, Scaleformer, Pathformer, and Itransformer) on the public datasets. Running the script "Run_SEMixer_Baselines_TSF.py" can reproduce the paper results of SEMixer and baselines.
  
