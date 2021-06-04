# 车牌识别服务端



## 1. Yolo车牌检测模型

1. 本项目使用darknet训练车牌检测模型。
2.  YoLo目标检测算法[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)
3. 本项目是已经编译好的darknet，直接跳转至[数据集](#dataset)开始做，<font color=#FF0000 size=4>**建议重新编译**</font>，请看第四条。
4. 若需要**重新编译**，跳转至[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)，编译完成之后，从[数据集](#dataset)开始做。

### 1.1. Requirements

* **CMake >= 3.18**: https://cmake.org/download/
* **Powershell** (already installed on windows): https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell
* **CUDA >= 10.2**: https://developer.nvidia.com/cuda-toolkit-archive 
* **OpenCV >= 2.4**: [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
* **cuDNN >= 8.0.2** https://developer.nvidia.com/rdp/cudnn-archive ，on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )

* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported##### 

### 1.2. compile on Windows (using `CMake`)

Requires:

* **MSVC**：https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community
* **CMake GUI**： `Windows win64-x64 Installer`https://cmake.org/download/
* **cmd**：`git clone git@github.com:AlexeyAB/darknet.git`

In Windows:

* Start (button) -> All programs -> CMake -> CMake (gui) ->

* [look at image](https://habrastorage.org/webt/pz/s1/uu/pzs1uu4heb7vflfcjqn-lxy-aqu.jpeg) In CMake: Enter input path to the darknet Source, and output path to the Binaries -> Configure (button) -> Optional platform for generator: `x64`  -> Finish -> Generate -> Open Project ->

* in MS Visual Studio: Select: x64 and Release -> Build -> Build solution

* find the executable file `darknet.exe` in the output path to the binaries you specified

![x64 and Release](https://habrastorage.org/webt/ay/ty/f-/aytyf-8bufe7q-16yoecommlwys.jpeg)

* 编译之后，把`\Release\darknet.ext`复制到`\build\darknet\x64\`文件夹下，双击不出现错误表明编译成功

### 1.3. 数据集<span id="dataset"> </span>

下述的存放路径都是在已经编译好的`darknet`项目文件夹下，即`darknet\build\darknet\x64\***`等。

存放路径：`\build\darknet\x64\data`，`\build\darknet\x64\data`

* 数据集来源：[YoLo数据集 ](https://gitee.com/lx1318753541/yolo-dataset)

* 数据集解释：

![gYA7V0.png](https://z3.ax1x.com/2021/05/09/gYA7V0.png)

每张车牌的`.jpg`图片对应一个`.txt`文件，**例如**

![gYA5Ks.jpg](https://z3.ax1x.com/2021/05/09/gYA5Ks.jpg)

对应的`.txt`文件的内容是

> 0 0.5091863517060368 0.7595238095238095 0.48293963254593175 0.36666666666666664

第一个数字0表示**第0类**（本数据集只设置一个车牌类别所以只有0）

第二个数字表示**中心点横坐标**

第三个数字表示**中心点纵坐标**

第四个数字表示**标注框宽度**（相对宽度）

第五个数字表示**标注框高度**（相对高度）

### 1.4. 训练所需其他文件

`train.txt`：train_images中所有图片的路径，每个图片一行，`build\darknet\x64\data`       [Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1UrqzksZ4Pt3cf8UQGTaoqbcZBN0ipnMg/view?usp=sharing)

`val.txt`：val_images中所有图片的路径，每个图片一行，`build\darknet\x64\data `     [Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1ZVRKGz4deIihBRLaOAwjkKWLXUY2FwVh/view?usp=sharing)

`KD.names`：本文件每一行是一个类名，本项目只有一类`plate`，`build\darknet\x64\data`      [Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1f54DGQQ_fCkvbXD7F1JcyVzl9b5TDjKa/view?usp=sharing)

`KD.data`：`build\darknet\x64\data`      [Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1iAFq8RXdTfkGuUrevovTZVyVOtObou5c/view?usp=sharing)

> classes= 1                   # 本项目设置只包含车牌一个种类 
> train = data/train.txt
> valid = data/val.txt
> names = data/KD.names
> backup = backup/      # 生成的模型保存的位置

`yolov3-KD.cfg`：复制`build\darknet\x64\cfg\yolov3.cfg`的内容修改其中的几处内容   [Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1jhwOxiwV5Qa3nEHvyTi9a98Zvq7w6Ho5/view?usp=sharing)

1. 每次计算的图片数目 = batch/subdivisions，GPU显存小的可以将batch调低
2. classes设置为1
3. filters设置为18，filters的计算公式为$filter = (classes + 5) * 3$

### 1.5. 预训练权重

存放路径：`build\darknet\x64`

[下载链接](https://pjreddie.com/media/files/darknet53.conv.74)：可以帮助训练更好的收敛

### 1.6. 开始训练<span id="start_train"> </span>

`darknet.exe detector train data/KD.data cfg/yolov3-KD.cfg darknet53.conv.74`

每迭代100次就会在backup文件夹上生成一个模型权重

最终使用的权重[Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1Lqcq_oA74vw5lh0_IyEi15eog-qE34Fm/view?usp=sharing)

### 1.7. 测试

`darknet.exe detector test data/KD.data cfg/yolov3-KD_test.cfg backup/yolov3-KD_last.weights data/val_images/4887.jpg -thresh 0.5`

> 注：`yolov3-KD_test.cfg`等于`yolov3-KD.cfg`，其中`batch`和`subdivisions`为1

![gYAHaV.png](https://z3.ax1x.com/2021/05/09/gYAHaV.png)![gYAobq.png](https://z3.ax1x.com/2021/05/09/gYAobq.png)![gYKJht.png](https://z3.ax1x.com/2021/05/09/gYKJht.png)![gYAIrn.png](https://z3.ax1x.com/2021/05/09/gYAIrn.png)



## 2. Pytorch训练LPRNET模型

### 2.1. Requirements

* pytorch==1.8.1

* cuda==10.1.168
* cudnn==7.6.5
* python==3.7
* opencv-python=4.5
* jupyter notebook
* numpy

### 2.2. 数据集

* 训练数据集来源：来自CCPD，取其中20%做训练集，10%做测试集
* 训练数据集：[Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1jF1I0I5ZCPXYlv0KdA5uZrtEoWhxWQ5A/view?usp=sharing)  
* 单张测试文件：[Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1iO_pi7opelkd6zoPi974hhk6bIFKOkc6/view?usp=sharing)  
* 多张测试文件：[Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1WAVTzvLg6hmr4NXvYsde6pgIi99unIOH/view?usp=sharing)

### 2.3. 训练

* 做30epoch的训练结果

![gYdvNV.png](https://z3.ax1x.com/2021/05/09/gYdvNV.png)

### 2.4. 模型测试结果

> ```
> Validation Accuracy: 0.8641534901658311 [26889:1240:2987:31116]
> ```

### 2.5. 模型下载链接

[Google Drive (访问需要挂梯子)](https://drive.google.com/file/d/1MOhUsgZ-ocx04ogWGWPzRvkW4JvBC3Mj/view?usp=sharing)

## 3. 部署至服务器

### 3.1 Django框架

Django 是一个开放源代码的 Web 应用框架，由 Python 写成。

### 3.2 Requirements

* Django=2.2.5
* opencv-python=4.5
* torch==1.8.1
* torchaudio==0.8.1
* torchvision==0.9.1

### 3.3 介绍

使用Django框架部署至服务端，开设端口，端口号为`139.196.240.235:10000`，客户端朝此ip地址发送图像的base64编码之后，服务端会解码，然后调用LPRNet进行识别，然后将识别成功的字符进行返回。

