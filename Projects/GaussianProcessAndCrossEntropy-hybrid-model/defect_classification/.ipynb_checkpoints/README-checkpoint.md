# DeepGD3_Plus

We present our code on PCB Dataset and Fruit Dataset in two folder. Please go into the links below for more information.

## Dataset

[**PCB Dataset**](https://github.com/jason211346/Phison-defect-detection-GP/tree/main/PCB_dataset) |  [**Fruit Dataset**](https://github.com/jason211346/Phison-defect-detection-GP/tree/main/Fruit_dataset)
:-------------------------:|:-------------------------:
<img src="img/PCB_dataset.png" alt="drawing" width="400"/> |  <img src="img/fruit_dataset.png" alt="drawing" width="400"/>

## Challenge

Defect detection is a common task in deep learning and is widely used in industrial image analysis. In practical applications, we define components with sufficient normal and defective samples as "old components", while components with insufficient defective samples or even a complete lack of defective samples as "new components". The defect types for both new and old components are the same, as shown in the table below. 

|         | Normal Samples | Defective Samples |
|---------|----------------|------------------|
| Old Component | Many           | Many             |
| New Component | Many           | Few or even none |

We found that general models perform poorly in detecting defects in new components. Therefore, we started to study how to design a model architecture that can better generalize defect features using only defective samples from old components when there is a lack of defective samples for new components in the dataset. The goal is to further improve the ability to detect defects in new components.

## Model architecture

- **Hybrid Expert Training :**
<img src="img/HBE.png" alt="drawing" width="900"/>

- **Hybrid Expert inference :**
<img src="img/HBE_inference.png" alt="drawing" width="900"/>

## Key technology
- **Imblanced sample in Component:**
- [x] Class Balanced Sampling : Class Balanced Sampling can keep the original data distribution and sample the same number of instances for each class during each sampling iteration. This can avoid model bias caused by imbalanced samples.
- **Defect Detector:**
- [x] CrossEntropy : Use CrossEntropy as the loss function for the defect detector.
- **Component Embedding:**
- [x] Gaussian Process : Gaussian Process can be used to learn the similarity between components and embed them into a high-dimensional vector space.

- **Inference:**
- [x] MLP classifier 𝜶 : A fully connected layer with 2 output dimension for good and bad classification.
- [x] Gaussian Process classifier 𝛃 : Gaussian Process classifier (GPC) for Component classification.
- [x] Prediction Converter 𝜸 : analyze the output U for good, bad, unknown classification
- [x] Prediction Combiner 𝐆 : combine the prediction result from 𝛂 and 𝛃 and yield the final prediction.
## Requirement
```
# 將Python升級到3.8: https://tech.serhatteker.com/post/2019-12/upgrade-python38-on-ubuntu/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U scikit-learn
pip install tensorboard tensorboardX selenium twder beautifulsoup4 seaborn thop tqdm pytorch_metric_learning openpyxl natsort tensorboard_logger opencv-python pandas seaborn 
pip install numpy==1.23.5
pip install bayesian-optimization==1.4.2
pip install pycave==3.1.3
pip install ipykernel --upgrade
python3 -m ipykernel install --user
pip install gpytorch==1.8.1
pip install pytorch-ignite
pip uninstall horovod -y
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
pip uninstall pandas 
pip install pandas==1.5.3
```
