# Local installation

The following steps describe how to install a CPU-based version
of [detectron2](https://github.com/facebookresearch/detectron2) on a local machine for development 
([based on this](https://detectron2.readthedocs.io/en/v0.5/tutorials/install.html)).

* create virtual environment

  ```commandline
  virtualenv -p /usr/bin/python3.7 venv
  ```
  
* install detectron2 

  * open *install* arrow on installation page for desired PyTorch version 
  
    https://detectron2.readthedocs.io/en/v0.4/tutorials/install.html#install-pre-built-detectron2-linux-only
    
  * open website in browser, e.g., for PyTorch 1.8:

    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
    
  * copy link, e.g., for Python 3.7:
  
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/detectron2-0.4%2Bcpu-cp37-cp37m-linux_x86_64.whl
    
  * install wheel file:
  
    ```commandline
    ./venv/bin/pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/detectron2-0.5%2Bcpu-cp37-cp37m-linux_x86_64.whl
    ```
    
  * install PyTorch
  
    ```commandline
    ./venv/bin/pip install torch==1.8.0
    ```

  * install torchvision (see [here](https://pypi.org/project/torchvision/) for version matrix):
  
    ```commandline
    ./venv/bin/pip install torchvision==0.9.0
    ``` 
    
  * install ONNX
  
    ```commandline
    ./venv/bin/pip install onnx
    ```
    