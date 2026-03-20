##### 原始

conda env export

```
(py8) PS D:\Work\yolo\NOW\YOLOv8-DeepSORT-Object-Tracking> conda env export
name: py8
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - defaults
dependencies:
  - bzip2=1.0.8=h0ad9c76_9
  - ca-certificates=2026.2.25=h4c7d964_0
  - libffi=3.5.2=h3d046cb_0
  - liblzma=5.8.2=hfd05255_0
  - liblzma-devel=5.8.2=hfd05255_0
  - libsqlite=3.52.0=hf5d6505_0
  - libzlib=1.3.1=h2466b09_2
  - openssl=3.6.1=hf411b9b_1
  - pip=24.3.1=pyh8b19718_0
  - python=3.8.20=hfaddaf0_2_cpython
  - setuptools=75.3.0=pyhd8ed1ab_0
  - tk=8.6.13=h6ed50ae_3
  - ucrt=10.0.26100.0=h57928b3_0
  - vc=14.3=h41ae7f8_34
  - vc14_runtime=14.44.35208=h818238b_34
  - vcomp14=14.44.35208=h818238b_34
  - wheel=0.45.1=pyhd8ed1ab_0
  - xz=5.8.2=hb6c8415_0
  - xz-tools=5.8.2=hfd05255_0
  - pip:
      - absl-py==2.3.1
      - antlr4-python3-runtime==4.9.3
      - asttokens==3.0.1
      - backcall==0.2.0
      - certifi==2026.2.25
      - cffi==1.17.1
      - charset-normalizer==3.4.6
      - colorama==0.4.6
      - contourpy==1.1.1
      - cryptography==46.0.5
      - cycler==0.12.1
      - decorator==5.2.1
      - easydict==1.13
      - executing==2.2.1
      - filelock==3.16.1
      - fonttools==4.57.0
      - fsspec==2025.3.0
      - gitdb==4.0.12
      - gitpython==3.1.46
      - google-auth==2.49.1
      - google-auth-oauthlib==1.0.0
      - grpcio==1.70.0
      - hydra-core==1.3.2
      - idna==3.11
      - importlib-metadata==8.5.0
      - importlib-resources==6.4.5
      - ipython==8.12.3
      - jedi==0.19.2
      - jinja2==3.1.6
      - kiwisolver==1.4.7
      - markdown==3.7
      - markupsafe==2.1.5
      - matplotlib==3.7.5
      - matplotlib-inline==0.1.7
      - mpmath==1.3.0
      - networkx==3.1
      - numpy==1.24.4
      - oauthlib==3.3.1
      - omegaconf==2.3.0
      - opencv-python==4.13.0.92
      - packaging==26.0
      - pandas==2.0.3
      - parso==0.8.6
      - pickleshare==0.7.5
      - pillow==9.5.0
      - polars==1.8.2
      - prompt-toolkit==3.0.52
      - protobuf==5.29.6
      - psutil==7.2.2
      - pure-eval==0.2.3
      - pyasn1==0.6.3
      - pyasn1-modules==0.4.2
      - pycparser==2.23
      - pygments==2.19.2
      - pyparsing==3.1.4
      - python-dateutil==2.9.0.post0
      - pytz==2026.1.post1
      - pyyaml==6.0.3
      - requests==2.32.4
      - requests-oauthlib==2.0.0
      - scipy==1.10.1
      - seaborn==0.13.2
      - six==1.17.0
      - smmap==5.0.3
      - stack-data==0.6.3
      - sympy==1.13.3
      - tensorboard==2.14.0
      - tensorboard-data-server==0.7.2
      - thop==0.1.1-2209072238
      - torch==2.4.1+cu118
      - torchaudio==2.4.1+cu118
      - torchvision==0.19.1+cu118
      - tqdm==4.67.3
      - traitlets==5.14.3
      - typing-extensions==4.13.2
      - tzdata==2025.3
      - ultralytics==8.4.23
      - ultralytics-thop==2.0.18
      - urllib3==2.2.3
      - wcwidth==0.6.0
      - werkzeug==3.0.6
      - zipp==3.20.2
prefix: D:\Anoconda\envs\py8

```

pip freeze

```
(py8) PS D:\Work\yolo\NOW\YOLOv8-DeepSORT-Object-Tracking> pip freeze
absl-py==2.3.1
antlr4-python3-runtime==4.9.3
asttokens==3.0.1
backcall==0.2.0
certifi==2026.2.25
cffi==1.17.1
charset-normalizer==3.4.6
colorama==0.4.6
contourpy==1.1.1
cryptography==46.0.5
cycler==0.12.1
decorator==5.2.1
easydict==1.13
executing==2.2.1
filelock==3.16.1
fonttools==4.57.0
fsspec==2025.3.0
gitdb==4.0.12
GitPython==3.1.46
google-auth==2.49.1
google-auth-oauthlib==1.0.0
grpcio==1.70.0
hydra-core==1.3.2
idna==3.11
importlib_metadata==8.5.0
importlib_resources==6.4.5
ipython==8.12.3
jedi==0.19.2
Jinja2==3.1.6
kiwisolver==1.4.7
Markdown==3.7
MarkupSafe==2.1.5
matplotlib==3.7.5
matplotlib-inline==0.1.7
mpmath==1.3.0
networkx==3.1
numpy==1.24.4
oauthlib==3.3.1
omegaconf==2.3.0
opencv-python==4.13.0.92
packaging==26.0
pandas==2.0.3
parso==0.8.6
pickleshare==0.7.5
Pillow==9.5.0
polars==1.8.2
prompt_toolkit==3.0.52
protobuf==5.29.6
psutil==7.2.2
pure_eval==0.2.3
pyasn1==0.6.3
pyasn1_modules==0.4.2
pycparser==2.23
Pygments==2.19.2
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2026.1.post1
PyYAML==6.0.3
requests==2.32.4
requests-oauthlib==2.0.0
scipy==1.10.1
seaborn==0.13.2
six==1.17.0
smmap==5.0.3
stack-data==0.6.3
sympy==1.13.3
tensorboard==2.14.0
tensorboard-data-server==0.7.2
thop==0.1.1.post2209072238
torch==2.4.1+cu118
torchaudio==2.4.1+cu118
torchvision==0.19.1+cu118
tqdm==4.67.3
traitlets==5.14.3
typing_extensions==4.13.2
tzdata==2025.3
ultralytics==8.4.23
ultralytics-thop==2.0.18
urllib3==2.2.3
wcwidth==0.6.0
Werkzeug==3.0.6
zipp==3.20.2

```



关于cuda

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```



py v 3.8.20

```
(py8) PS D:\Work\yolo\NOW\YOLOv8-DeepSORT-Object-Tracking> python --version
Python 3.8.20

```



##### 迁移

Python 3.8.20

```
pip install --default-timeout=1000 torch==2.4.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
```

requirement.txt

```
absl-py==2.3.1
antlr4-python3-runtime==4.9.3
asttokens==3.0.1
backcall==0.2.0
certifi==2026.2.25
cffi==1.17.1
charset-normalizer==3.4.6
colorama==0.4.6
contourpy==1.1.1
cryptography==46.0.5
cycler==0.12.1
decorator==5.2.1
easydict==1.13
executing==2.2.1
filelock==3.16.1
fonttools==4.57.0
fsspec==2025.3.0
gitdb==4.0.12
GitPython==3.1.46
google-auth==2.49.1
google-auth-oauthlib==1.0.0
grpcio==1.70.0
hydra-core==1.3.2
idna==3.11
importlib_metadata==8.5.0
importlib_resources==6.4.5
ipython==8.12.3
jedi==0.19.2
Jinja2==3.1.6
kiwisolver==1.4.7
Markdown==3.7
MarkupSafe==2.1.5
matplotlib==3.7.5
matplotlib-inline==0.1.7
mpmath==1.3.0
networkx==3.1
numpy==1.24.4
oauthlib==3.3.1
omegaconf==2.3.0
opencv-python==4.13.0.92
packaging==26.0
pandas==2.0.3
parso==0.8.6
pickleshare==0.7.5
Pillow==9.5.0
polars==1.8.2
prompt_toolkit==3.0.52
protobuf==5.29.6
psutil==7.2.2
pure_eval==0.2.3
pyasn1==0.6.3
pyasn1_modules==0.4.2
pycparser==2.23
Pygments==2.19.2
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2026.1.post1
PyYAML==6.0.3
requests==2.32.4
requests-oauthlib==2.0.0
scipy==1.10.1
seaborn==0.13.2
six==1.17.0
smmap==5.0.3
stack-data==0.6.3
sympy==1.13.3
tensorboard==2.14.0
tensorboard-data-server==0.7.2
thop==0.1.1.post2209072238
torch==2.4.1+cu118
torchaudio==2.4.1+cu118
torchvision==0.19.1+cu118
tqdm==4.67.3
traitlets==5.14.3
typing_extensions==4.13.2
tzdata==2025.3
ultralytics==8.4.23
ultralytics-thop==2.0.18
urllib3==2.2.3
wcwidth==0.6.0
Werkzeug==3.0.6
zipp==3.20.2
```

```
pip install -r requirements.txt
```

