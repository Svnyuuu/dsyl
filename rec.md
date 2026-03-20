#### TASK

```
视频 → YOLO检测 → DeepSORT跟踪 → 轨迹
                                ↓
                ┌───────────────┼───────────────┐
                ↓               ↓               ↓
            计数模块        越界检测        评估模块
```



​	✅️trackvid读懂  

​	修改右上错误框选区（暂时不需要 毕竟已经会了禁用框选区 之后再加上就好了

​	✅️好像是没加deepsort的 给他加上！（deepsort是用于追踪的

​	（加上了但是感觉和原本的yolo track大差不差 甚至更差 所以直接用的别人的开源项目 原本的yolo可以当成对比数据



​	如何 “ 与传统计数方法（如背景建模）以及其他主流跟踪算法（如ByteTrack）进行对比，通过检测精度（mAP）、跟踪精度（MOTA）、计数准确率（MAE）等指标验证本系统的优越性。

​	虚拟线计数、禁区判定  实现实时统计与报警

​	检测性能 COCO API进行评估

​	跟踪性能借助MOT Challenge官方工具

​	计数与越界效果则通过自编脚本计算准确率与召回率



说是

```
检测性能

	mAP（COCO）

跟踪性能

	MOTA（MOT Challenge）

计数性能

	MAE

行为检测

	Precision / Recall（越界）
```

​	flask并非实时输出 等代码完全弄好了就重新封装一个web



​	或许按着大纲开始写论文？



#### ques+ppt

https://orcapaper.cn/knowledge/thesis-guide/opening-report-defense-guide-preparation-strategies



#### 关键函数

ref

```
https://www.runoob.com/opencv/opencv-video.html
```

| 参数                         | 属性常量 | 说明                                 |
| :--------------------------- | :------- | :----------------------------------- |
| `cv2.CAP_PROP_POS_MSEC`      | 0        | 视频文件的当前位置（毫秒）           |
| `cv2.CAP_PROP_POS_FRAMES`    | 1        | 基于0开始的帧索引位置                |
| `cv2.CAP_PROP_POS_AVI_RATIO` | 2        | 视频文件的相对位置（0=开始，1=结束） |
| `cv2.CAP_PROP_FRAME_WIDTH`   | 3        | 视频流中**帧的宽度**                 |
| `cv2.CAP_PROP_FRAME_HEIGHT`  | 4        | 视频流中**帧的高度**                 |
| `cv2.CAP_PROP_FPS`           | 5        | 视频的**帧速率**（帧/秒）            |

track official doc

```
https://docs.ultralytics.com/modes/track/#python-examples
```



#### 增加deepsort

```
pip install deep-sort-realtime
```

```
DeepSORT（Deep Learning + SORT）是一种基于深度学习和轨迹排序的多目标追踪算法。它在SORT（Simple Online and Realtime Tracking）算法的基础上引入了卷积神经网络（CNN）来提取目标特征并实现更精确的目标关联。
   
所以DeepSort的前身是Sort算法，而Sort的核心是基于卡尔曼滤波和匈牙利算法，通过预测和关联匹配来跟踪目标。
```





#### lyg

```bash
pip install -r requirements.txt
```

```bash
pip install ultralytics
```

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -m ultralytics.yolo.v8.detect.predict model=yolov8l.pt source="test3.mp4" show=True device=0

```bash
python -m ultralytics.yolo.v8.detect.predict model=yolov8l.pt source="test3.mp4" show=True device=cuda
```

```bash
python predict.py model=yolov8l.pt source="test3.mp4" show=True
```

```bash
cd D:\Work\yolo\NOW\YOLOv8-DeepSORT-Object-Tracking
set PYTHONPATH=D:\Work\yolo\NOW\YOLOv8-DeepSORT-Object-Tracking
python ultralytics/yolo/v8/detect/predict.py model=yolov8l.pt source="test3.mp4" show=True
```

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics
```

```bash
# 解决上面network timeout
pip install --default-timeout=1000 torch==2.4.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118        
```



###### 开源 ultralytics

```
https://github.com/ultralytics/
```



#### YOLOv8-DeepSORT-Object-Tracking

以下可以运行成功

py8环境

```bash
cd D:\Work\yolo\NOW\YOLOv8-DeepSORT-Object-Tracking

python -m ultralytics.yolo.v8.detect.predict model=yolov8l.pt source="test3.mp4" show=True
```

```bash
python -m ultralytics.yolo.v8.detect.tracking model=yolov8l.pt source="test3.mp4" show=True
```

```bash
python -m ultralytics.yolo.v8.detect.predict_div model=yolov8l.pt source="test3.mp4" show=True
```

使用gpu跑👇（用的torch

```bash
python -m ultralytics.yolo.v8.detect.predict_div model=yolov8l.pt source="test3.mp4" show=True device=0
```



##### predict

​	用于标记id和分类的视频处理和输出

##### val

​	用来评估结果

​	（P R mAP50）prediction recall IoU=50的平均精度

#### 改动

###### pillow版本不兼容

```
AttributeError: 'FreeTypeFont' object has no attribute 'getsize'
```

​	查看原本版本为

```
pip pillow --version
Version: 10.4.0
```

​	降级

```
pip install Pillow==9.5.0
```



#### 需要增加的

###### ✅️分界线流入流出计数  速度计算（km/h）左右方向计数（流入 / 流出）

​	实则左右计数反了没改暂时 但是可以正常计数分界线流入流出

```
ultralytics/yolo/engine/predictor.py
```

```python
# 增加了
def show(self, p):
    if self.annotator is not None:
        im0 = self.annotator.result()
    else:
        im0 = self.plotted_img   # ⭐ 用你自己画的图

    cv2.imshow(str(p), im0)
    cv2.waitKey(1)
```

```python
# 修改了
if self.annotator is not None:
    im0 = self.annotator.result()
else:
    im0 = self.plotted_img
```



###### ✅️把cpu改成gpu

原始

```
Ultralytics YOLOv8.0.3  Python-3.10.20 torch-2.10.0+cpu CPU
```

```bash
# 选择安装了
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
python -m ultralytics.yolo.v8.detect.predict_div model=yolov8l.pt source="test3.mp4" show=True device=0
```



###### ⚠跳变

​	有时候经常在两个cls里面跳变（依然truck和car

###### ⚠truck和car经常识别混淆

###### ✅️刚进入区域的框选区容易很大 

​	在车辆完全进入画面会恢复正常框选区 直接通过限定测试区解决了

###### **增加感兴趣区域（ROI，Region of Interest）**的过滤功能 

###### ✅️添加**区域限制**功能，只统计**机动车道内**的车辆，并过滤掉**非车辆的误识别**

​	只在特定区域识别车辆（因为区域外并无车 不显示区域外的车（不进行识别

###### ✅️测试的换成了yolov8m（差别不大

###### ⚠predict_div

​	检测人有问题啊

​	而且行人车辆视频不算好找

​	直接先行人车辆分开做了

​	或者先只做车

​	待定是纯行人的增加禁止区域显示warning

#### 过程数据

train运行之后的评估（部分

![image-20260318190023745](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20260318190023745.png)



test1看起来还可以 部分识别成摩托车或许可以限定只显示行人+bus truck之类的（没找同时具有行人车辆的视频素材

test2测试是很糟糕的 很多truck和car都混淆了

test3即开源给的视频素材 检测一般般吧就那样
