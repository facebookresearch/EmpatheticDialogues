## setup steps

We will setup two conda environments, one for running the code in the original paper and the other for running the code in the new environment.

> Environment 1 : empathy01

`conda create -n empathy01 python=3.6`

`conda activate empathy01`

`conda install numpy=1.14.3`

`conda install pytorch-cpu==1.0.1 torchvision-cpu==0.2.2 cpuonly -c pytorch`

`pip install tqdm==4.19.7`

`conda install pandas=0.22.0`

`pip install fairseq==0.6.2`

`pip install git+git://github.com/facebookresearch/ParlAI.git@471db18c47d322d814f4e1bba6e35d9da6ac31ff`

`pip install pytorch-pretrained-bert==0.5.1`

Set environment variable to select EMO classifier model `EMO_MODEL=fast(default)|lstm|attn|trans`

> Environment 2 : empathy02

`pip install tqdm==4.19.7`

`pip install tensorflow==2.4.1`

`pip install keras==2.4.3`

`pip install pytorch-pretrained-bert==0.5.1`

`pip install transformers==4.12.5`

## code structure

Apart from the original code, the files that we introduced reside in `empchat/classifiers` folder and `data` folder.

`empchat/classifiers` folder contains the model training and evaluation files for the Emotion classifiers.

## running



