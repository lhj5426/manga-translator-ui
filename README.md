为使用自训练模型

https://github.com/lhj5426/YSG

作为主检测而按照自身喜好魔改的版本

仅根据个人在window10 系统上的使用习惯而魔改

不管其他系统环境下的问题

只专注于window系统上的使用体验

自用版

检测需要使用数据标注工具

https://github.com/lhj5426/X-AnyLabeling

加载自训练模型

https://github.com/lhj5426/YSG

 导出YOLO格式标注数据
 
 <img width="1017" height="371" alt="image" src="https://github.com/user-attachments/assets/2dee05ec-4898-4aa6-94b1-750b7f2e5d98" />

通过导入数据使用

或者
直接在标注软件上导出

<img width="628" height="690" alt="image" src="https://github.com/user-attachments/assets/a51abf77-43d2-452e-b2f2-040d383b4597" />

来执行后续操作

<img width="1038" height="357" alt="image" src="https://github.com/user-attachments/assets/c4759b42-94b3-44ba-872d-ac42eeaa2322" />

 小众用法 仅供个人使用 

 并且可以使用

 https://github.com/lhj5426/YSG/blob/main/%E6%BC%AB%E7%94%BB%E8%BD%AF%E4%BB%B6/%E6%8B%96%E6%8B%BDmtuJSON%E6%96%87%E4%BB%B6%E5%A4%B9%E8%BD%ACBallonsTranslator%E5%8D%95JSON%E6%96%87%E4%BB%B6.py

 https://github.com/lhj5426/YSG/blob/main/%E6%BC%AB%E7%94%BB%E8%BD%AF%E4%BB%B6/%E6%8B%96%E6%8B%BDBallonsTranslator%E5%8D%95JSON%E8%BD%ACMTUjson%E9%A1%B9%E7%9B%AE%E6%96%87%E4%BB%B6%E5%A4%B9.py

 脚本 转换成

 https://github.com/dmMaze/BallonsTranslator

 可用的格式 实现2个软件的 无缝切换



https://anaconda.org/channels/conda-forge/packages/pydensecrf/overview
安装

查看现有环境
conda env list
.创建python环境
conda create -n MTUI python=3.12 -y

查看现有环境
conda env list
 激活该环境
conda activate MTUI

安装

conda install pydensecrf-1.0rc3-py312h72972c8_6.conda
https://anaconda.org/channels/conda-forge/packages/pydensecrf/overview
pip uninstall numpy -y
python -c "import pydensecrf; print('pydensecrf OK')"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_gpu_cu118.txt
python -c "import numpy; print(numpy.__version__)"
python -c "import numpy; import torch; print(numpy.__version__, torch.__version__)"
python 检测是否能运行GPU.py
启动
python desktop_qt_ui\main.py
