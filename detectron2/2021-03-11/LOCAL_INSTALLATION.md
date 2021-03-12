# Local installation

The following steps describe how to install a CPU-based version
of [detectron2](https://github.com/facebookresearch/detectron2) on a local machine for development 
([based on this](https://detectron2.readthedocs.io/en/v0.3/tutorials/install.html)).

* create virtual environment

  ```commandline
  virtualenv -p /usr/bin/python3.7 venv
  ```
    
* install PyTorch
  
  ```commandline
  ./venv/bin/pip install torch==1.7.1
  ```

* install torchvision (see [here](https://pypi.org/project/torchvision/) for version matrix):
  
  ```commandline
  ./venv/bin/pip install torchvision==0.8.2
  ``` 
  
* install detectron2 

  ```commandline
  ./venv/bin/pip install git+git://github.com/facebookresearch/detectron2.git@083a70b98959f59cd9ec6960fabd655deaabd742

  ```
  