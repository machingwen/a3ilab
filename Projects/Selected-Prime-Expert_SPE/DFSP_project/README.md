# Decomposed Fusion with Soft Prompt (DFSP)
DFSP is a model which decomposes the prompt language feature into state feature and object feature, then fuses them with image feature to improve the response for state and object respectively.


## block diagram
-baseline model
<img src="readme_/baseline model.png" alt="drawing" width="900"/>

-Our Model linear projection Architecture
<img src="readme_/Our Model linear projection Architecture.png" alt="drawing" width="900"/>


If you already have setup the datasets, you can use symlink and ensure the following paths exist:
`data/<dataset>` where `<datasets> = {'mit-states', 'ut-zappos', 'cgqa'}`.



## Challenge
The goal of this paper is to conduct experiments on the CZSL dataset to predict the attributes of an object given its known situation. By integrating the expert1 and expert2 models, we aim to reduce the Uncertainty Calibration Error (UCE) of the model and improve its accuracy.

In this study, we also used different evaluation metrics to test the performance of the model and demonstrated that integrating expert1 and expert2 can achieve better performance on the CZSL dataset by comparing different model configurations with different integration methods.




## Setup
```
conda create --name clip python=3.7
conda activate clip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install git+https://github.com/openai/CLIP.git
```
Alternatively, you can use `pip install -r requirements.txt` to install all the dependencies.

## Download Dataset
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.
```
sh download_data.sh
```


## Training
```
python -u train.py --dataset <dataset>
```
## Evaluation
We evaluate our models in two settings: closed-world and open-world.

### Closed-World Evaluation
```
python -u test.py --dataset <dataset>
```
You can replace `--dataset` with `{mit-states, ut-zappos}`.


### Open-World Evaluation
For our open-world evaluation, we compute the feasbility calibration and then evaluate on the dataset.

### Feasibility Calibration
We use GloVe embeddings to compute the similarities between objects and attributes.
Download the GloVe embeddings in the `data` directory:

```
cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
```
Move `glove.6B.300d.txt` into `data/glove.6B.300d.txt`.

To compute feasibility calibration for each dataset, run the following command:
```
python -u feasibility.py --dataset mit-states
```
The feasibility similarities are saved at `data/feasibility_<dataset>.pt`.
To run, just edit the open-world parameter in `config/<dataset>.yml`

### model 

https://huggingface.co/SerenityYuki/CZSL/tree/main
https://huggingface.co/SerenityYuki/CZSL-mit/tree/main
### SPE and HMOE 
```
python -u test_v4.py --dataset config1_<dataset>
```
To run, just edit the open-world parameter in `config1_<dataset>.yml`

    
## References
```
@article{lu2022decomposed,
  title={Decomposed Soft Prompt Guided Fusion Enhancing for Compositional Zero-Shot Learning},
  author={Lu, Xiaocheng and Liu, Ziming and Guo, Song and Guo, Jingcai},
  journal={arXiv preprint arXiv:2211.10681},
  year={2022}
}
```