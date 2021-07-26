# pytorch-test

Simple test script that builds an image classification model
using the CIFAR10 challenge data. Code based on pytorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Can be used to test a GPU machine from within a docker container.

## Test

* Assumptions

  * `/Scratch/fracpete` contains the `pytorchtest.py` script

* launch docker container

  ```commandline
  docker run \
     -u $(id -u):$(id -g) -e USER=$USER \
    --runtime=nvidia \
    --shm-size 8G \
    -v /Scratch/fracpete:/opt/projects \
    -it public.aml-repo.cms.waikato.ac.nz:443/pytorch/detectron2:0.3
  ```

* run the script

  ```commandline
  cd /opt/projects
  python pytorchtest.py
  ```
  
  **Notes:**

    * The first output should be `cuda:0` when the script runs on the GPU
    * 2 epochs of 12000 iterations should run

