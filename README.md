

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
### Imagenet with creating simulator and table
```
python3 quant_train.py -a mobilenetv2_100 --pretrained --epochs 80 --lr 1e-3 -b 256 --data <path to DS> -j 16 --save-path <save path> --wd 1e-5 -p 50 -qf 2 --beta 1 -EMA 0.01 --create_table --create_sim
```

defult HW arguments given in the code as
```
hw_args = ('eyeriss',12,14,108,108,108,'ws',10,1,200000000,True)
```



## Updated:

1. trained models will be updated soon


## Future work
1. add power estimation to all benchmarks
2. add support for other simulators
4. add CO2 consumption of our training procedure


## License
released under the [MIT license](LICENSE).
