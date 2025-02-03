# Python 3.8: https://tech.serhatteker.com/post/2019-12/upgrade-python38-on-ubuntu/
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
# cd ./clustimage_phison

pip install -r clustimage_phison/requirements.txt
# deepGD3_plus/defect_classification/PCB_dataset/clustimage_phison/requirements.txt