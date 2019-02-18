# The MXNet Implementation of ShuffleNet v1 and v2

This repository includes codes for ShuffleNet v1 and v2. In addition, MobileFaceNet, which is an efficient mobile CNN for face verification introduced in [arxiv](https://arxiv.org/abs/1804.07573) is also included in this repository.

## Environment

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (=1.3.0)

## Usage

To test ShuffleNet v1, v2 and MobileFaceNet on mnist
```
python test_mnist.py
```


## License

MIT LICENSE


## Reference

```
@article{Zhang2017ShuffleNet,
  title={ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices},
  author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Jian, Sun},
  year={2017},
}

@article{Ma2018ShuffleNet,
  title={ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai Tao and Sun, Jian},
  year={2018},
}

@article{Chen2018MobileFaceNets,
  title={MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices},
  author={Chen, Sheng and Liu, Yang and Gao, Xiang and Han, Zhen},
  year={2018},
}
```

## Acknowledgment

The code is adapted based on an intial fork from the [MXShuffleNet](https://github.com/ZiyueHuang/MXShuffleNet) repository.
