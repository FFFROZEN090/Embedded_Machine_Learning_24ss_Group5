FROM nvcr.io/nvidia/pytorch:21.08-py3
#first tries with 18.05-py3 
#latest image:21.06-py3
#19.05-py3 not compatible with gpu
#try tobis version
RUN set -ex

RUN pip install --upgrade pip && \
    pip install Cython tensorboard matplotlib numpy segmentation-models-pytorch opentsne seaborn scikit-learn scipy elasticdeform && \
    pip install pillow==9.5.0 scikit-image==0.21.0 imagecodecs==2022.12.24 opencv-python==4.1.2.30 opencv-python-headless==4.1.2.30 torchmetrics==1.2.0 deepcell==0.12.9