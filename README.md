# EasyAIVtuber

> 驱动你的纸片人老婆。
Simply animate your 2D waifu.

Fork自 [`yuyuyzl/EasyVtuber`](https://github.com/yuyuyzl/EasyVtuber)。由于是AI Vtuber，因此去掉了原项目的面捕功能。
本项目配合stable diffusion等文生图模型为最佳食用方式。

## Features not available in the original repo

1. 空闲自动做动作（眨眼、东张西望）
2. 说话动作（自动对口型）
3. 摇子（自动随节奏点头）
4. 唱歌动作（自动对口型，跟随节奏摇摆）(更新中... 0.0)
5. API调用接口

## Installation
### 安装依赖库
创建并激活虚拟环境  
```
conda create -n eaiv python=3.10
conda activate eaiv
```
安装torch（最好是30系显卡及以上）
```
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```
然后在项目目录下执行以下命令  
```
pip install -r requirements.txt
```

### 下载预训练模型  

https://www.dropbox.com/s/y7b8jl4n2euv8xe/talking-head-anime-3-models.zip?dl=0  
解压到`data/models`文件夹中，与`placeholder.txt`同级  
正确的目录层级为  
```
+ models
  - separable_float
  - separable_half
  - standard_float
  - standard_half
  - placeholder.txt
```
### 安装OBS
可在网上自行搜索教程安装

### 安装UnityCapture

> 注：如果电脑上安装过VTube Studio，也许OBS的视频采集设备的设备中就会有 VTubeStudioCam（没做过实验不太确定）。
> 若有此设备，便无需执行下面步骤安装UnityCapture，直接使用 VTubeStudioCam 即可

为了能够在OBS上看到纸片老婆并且使用透明背景，需要安装UnityCapture  
参考 https://github.com/schellingb/UnityCapture#installation  
只需要正常走完Install.bat，在OBS的视频采集设备中便能看到对应的设备（Unity Video Capture）。

#### 如何使背景透明
在OBS添加完视频采集设备以后，右键视频采集设备-设置-取消激活-分辨率类型选自定义-分辨率512x512(与`--output_size`参数一致)-视频格式选ARGB-激活

## Usage
1. 打开OBS并配置好视频采集设备
2. 将`main.bat`中第一行的虚拟环境的路径修改为你自己的虚拟环境路径
3. 运行`main.bat`
4. 使用post请求 http://127.0.0.1:7888/alive （默认端口为7888），
并传入相应参数便可使用音频文件自动生成模型的动作，具体示例可参考`test.py`

### 参数注解

|        参数名        |  值类型  |                                                                               说明                                                                                |
|:-----------------:|:-----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    --character    |  str  |                                                                `data/images`目录下的输入图像文件名，不需要带扩展名                                                                 |
|   --output_size   |  str  |                                                 格式为`512x512`，必须是4的倍数。<br>增大它并不会让图像更清晰，但配合extend_movement会增大可动范围                                                 |
|    --simplify     |  int  |                                                             可用值为`1` `2` `3` `4`，值越大CPU运算量越小，但动作精度越低                                                             |
|  --output_webcam  |  str  |                                                             可用值为`unitycapture`，选择对应的输出种类，不传不输出到摄像头                                                              |
|      --model      |  str  | 可用值为`standard_float` `standard_half` `separable_float` `separable_half`，<br/>其中standard_float占用最多显存效果最好，separable_half占用最少显存效果最逊（但也完全够用了），float为双精度，half为单精度模型， |
|  --port  |  int  |                                                                 本地API的端口号，默认为7888，若7888被占用则需要更改                                                                 |
| --extend_movement | float |                                                  （暂时没有用）根据头部位置，对模型输出图像进一步进行移动和旋转使得上半身可动<br>传入的数值表示移动倍率（建议值为1）                                                   |

