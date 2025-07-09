# 本分支已集成teacache

在cached_models文件夹的`google--gemma-2-2b`和`black-forest-labs--FLUX.1-dev`内，请去`https://huggingface.co/google/gemma-2-2b/tree/main`获取权限以后再把`model-00001-of-00003.safetensors`，`model-00002-of-00003.safetensors`，`model-00003-of-00003.safetensors`放到`google--gemma-2-2b`目录下，国内也可以在`https://www.modelscope.cn/models/google/gemma-2-2b/files`下载，会快不少，flux那个vae文件夹也要下，具体看下面的图

需要这些东西先下载好，cached_models文件夹如果没有就创一个
![image](https://github.com/user-attachments/assets/483e7095-0a4d-4872-8f01-5239a09a634c)
![image](https://github.com/user-attachments/assets/d75fdd28-41a1-4fc9-ba80-e8121abff386)
![image](https://github.com/user-attachments/assets/e4c4bed3-8f4b-4717-b90e-e04dafbaaf67)

dit的文件放到./ckpt目录下面
![image](https://github.com/user-attachments/assets/f8734a8e-9087-420e-9ca0-83a70b82ac4f)
注意那个model_args.pth别删

装好模型以后根目录下运行`python demo.py`有可选参数--port 6006这种
