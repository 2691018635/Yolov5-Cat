# Yolov5-Cat

使用[**Yolov5深度学习模型**](https://github.com/ultralytics/yolov5)，通过[**roboflow**](https://roboflow.com/)自建数据集的方式实现图片，视频和摄像机设备的猫咪检测。

### 运行环境

硬件环境：显卡 Nvidia RTX 3090 24G*2

操作系统：Windows 11专业版+Ubuntu 20.04

软件和主要packeage版本：Pycharm 2022.2 + **Python 3.9** + Anaconda + **PyTorch1.12.1-cu116**

### 数据集

图片来自Github上的开源数据集[**maxogden**/cats](https://github.com/maxogden/cats)，图片共195张照片。使用**roboflow**给图片中的猫咪手动打上标签，并将数据集分为train，test和valid三个子集，图片张数分别为137，20，38.


|原图片|打上标签后图片|
|:--:|:--:|
|<img src="https://s2.loli.net/2022/10/23/eivmwkbR1CY2l3X.jpg" alt="test1" style="zoom:20%;" />|<img src="https://s2.loli.net/2022/10/23/HEBMVvqFptA2w3Q.jpg" alt="test1" style="zoom:20%;" />|

### 模型配置

从Github上clone到本地后安装所需依赖

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

之后在Jupyter Notebook中执行一下代码从roboflow上下载处理后的数据集

```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="LjeDwpSQMHsBancGR1ii")
project = rf.workspace("northestern-university").project("nbqs")
dataset = project.version(1).download("yolov5")
```

### 参数选择

#### train.py

```python
#初始化权重，默认,选择了yolov5s模型，这是最简单的模型参数，训练也最快
parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
#配置文件，默认
parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
#数据集
parser.add_argument('--data', type=str, default=ROOT / 'NBQS-1/data.yaml', help='dataset.yaml path')
#超参数集，我们第一次训练使用模型自带的超参数集，不会影响训练结果
parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
#epoch，默认
parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
#batch-size，这个参数根据设备性能选定，在我的本地主机上只能设置为1...，如果多GPU运行则是所有GPU的batch-size
parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')
#图片大小，会对数据图片进行缩放
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--noval', action='store_true', help='only validate final epoch')
parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
parser.add_argument('--noplots', action='store_true', help='save no plot files')
parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
#GPU设备，我设置使用两个GPU，但好像最后也只使用了一个...应该是还需要对配置文件进行修改
parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
```

#### detect.py

```py
#权重选择，选择我们训练出的最好的权重
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp13/weights/best.pt', help='model path or triton URL')
#检测图片，可以说图片，视频和摄像头(参数设置为0就会调用设备的摄像头，然后将检测结果输出为一个mp4文件)
parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob/screen/0(webcam)')
#数据集
parser.add_argument('--data', type=str, default=ROOT / 'NBQS-1/data.yaml', help='(optional) dataset.yaml path')
```

**注：除了代码中标志的参数，其余参数都是用默认值**

### 模型效果

#### loss和一些其他参数随着epochs的变化

![results](https://s2.loli.net/2022/10/23/PsudHq4KVbJhBil.png)

#### 混淆矩阵

<img src="https://s2.loli.net/2022/10/23/RadpJbVeC15jnhD.png" alt="confusion_matrix" style="zoom:20%;" />

#### F1曲线

<img src="https://s2.loli.net/2022/10/23/fr2DxeCHUa7Opi6.png" alt="F1_curve" style="zoom:30%;" />

#### 部分训练结果

<img src="https://s2.loli.net/2022/10/23/n7osZhBkxcmFqO4.jpg" alt="train_batch0" style="zoom: 40%;" />

<center>train_batch0</center>

#### 检测结果

<img src="https://s2.loli.net/2022/10/23/HEBMVvqFptA2w3Q.jpg" alt="test1" style="zoom: 25%;" />

<center>单个目标检测</center>

<img src="https://s2.loli.net/2022/10/23/CYB72u5GnLrqoSp.jpg" alt="000122_ZV_SCSW_HeroImages_1900x1024_D1" style="zoom:40%;" />

<center>多个目标检测</center>

### 总结

从训练结果看模型的训练效果很好，各项评价指标都较好。但是在检测结果中，模型对于单目标的检测效果很好，但是遇到多目标问题是检测就会出现问题。初步认为是数据集中多目标数据较少，导致模型出现问题。

同时在手工处理数据集时，并没有严格按照物体的轮廓进行标志，所以可能对训练结果产生一定的影响，之后处理时会考虑到这个问题。

目前打算从两个方面改进，一个是丰富数据集。初步打算从视频中截取出图片来用于训练，这样不仅可以获得大量的数据，同时还能获得较多的多目标数据。第二个打算是区分猫的品种，这个目前也仅是个想法，打算先在数据处理时将不同品种的猫标记为不同的目标，可以先进行尝试看看效果。
