目前的两个尝试	

1，针对edsr的简单的多帧融合。模型文件./src/model/vedsr.py。改动了一点EDSR，主要想法是利用EDSR提取特征，多帧融合。在EDSR前加了运动估计（相当于图像校正，把前后帧校准为当前帧）。把warp过的图像和当前帧经过EDSR提取特征，经过一个注意力模块，用一层卷积融合，再经过5层残差网络，使用EDSR的解码的方法把特征解码为高帧率图像。分数提高的不明显可能和损失和feed数据的方式有关系。

2，参考inception结构改进的wdsr_b。模型文件./src/model/wdsr_inception.py.



**实验结果：**

| 模型                         | PSNR  |
| ---------------------------- | ----- |
| EDSR（单帧）                 | 37.99 |
| VEDSR（3帧融合，初赛提交的） | 38.40 |
| WDSR_b（单帧）               | 38.11 |
| WDSR_INCEPTION（单帧）       | 38.27 |

 	

### 环境依赖：

- CUDA 8.0.44
- CUDNN 6.0.21
- Python 3.5
- PyTorch >= 1.0.0
- numpy
- skimage
- imageio
- matplotlib
- tqdm

### 训练vedsr：

```
cd code/src/
```

冻结EDSR参数训练：

```
CUDA_VISIBLE_DEVICES=3 python main.py --data_train VSRData --data_test VSRData --scale 4 --pre_train download --batch_size 16 --n_GPUs 1 --save_model --model vedsr --epoch 30 --n_sequence 3 --lr 2e-4
```

微调：

```
CUDA_VISIBLE_DEVICES=3 python main.py --data_train VSRData --data_test VSRData --scale 4 --pre_train download --batch_size 16 --n_GPUs 1 --resume -1  --save_model --model vedsr  
```

### 测试vedsr：

测试并获取提交文件：

```
CUDA_VISIBLE_DEVICES=2 python main.py --data_test VSRDemo --test_only --scale 4 --resume -1  --save_results --model vedsr

cd ../experiment/test/

bash get_submit.sh
```



------



## 训练单帧模型（修改参数 --model）：

```
python main.py --data_train VSRData --data_test VSRData --scale 4 --batch_size 16 --n_GPUs 1 --save_model --model wdsr_inception --epoch 170 --n_sequence 1
```

## 测试集单帧模型：

```
python main.py --data_test VSRDemo --test_only --scale 4 --resume -1  --save_results --model wdsr_inception  
```
