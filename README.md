

# AMED: Automatic Mixed-Precision Quantization for Edge devices

This work uses [Scalesim](https://scalesim-project.github.io/) systolic array simulator 


## Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* **To install** and develop locally:
```bash
git clone https://github.com/RamorayDrake/AMED.git
cd AMED
pip install -r requirements.txt
```

## Getting Started
### Cifar100
```
python3 quant_train.py -a resnet18 --pretrained --epochs 50 --lr 0.001 -b 512 --ds cifar10 --data ./ --save-path checkpoints/ --wd 1e-4 -p 50 -qf 1 --create_table
```
### Imagenet
```
python3 quant_train.py -a resnet18 --pretrained --epochs 50 --lr 0.001 -b 128 --ds Imagenet --data PATH_TO_IMAGENET --save-path checkpoints/ --wd 1e-4 -p 50 -qf 1 --create_table
```

## Updated:

## Future work
1. add power estimation to all benchmarks
2. add support for other simulators
4. add CO2 consumption of our training procedure


## License
released under the [MIT license](LICENSE).
