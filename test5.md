# 实验五： 使用TensorFlow Lite Model Maker生成图像分类器模型

### 一、下载运行必备的一些库：

#### ①：下载 tflite-model-maker库 ：
    
    输入命令：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ tflite-model-maker，安装tflite-model-maker库
    
    ![image]( https://github.com/FurMax/AndroidTest2/blob/image/d1.png)

     ![image]( https://github.com/FurMax/AndroidTest2/blob/image/d2.png)

    下载完相关依赖后，发现报错，报错提示没有安装conda-repo-cli==1.0.4 和 anaconda-project==0.9.1

#### ②： 下载conda-repo-cli 1.0.4:

    输入命令： pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ conda-repo-cli==1.0.4 ,安装conda-repo-cli==1.0.4
        ![image]( https://github.com/FurMax/AndroidTest2/blob/image/d3.png)
        ![image]( https://github.com/FurMax/AndroidTest2/blob/image/d4.png)


    
#### ③： 下载anaconda-project==0.9.1

    输入命令: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ anaconda-project==0.9.1
        ![image] (https://github.com/FurMax/AndroidTest2/blob/image/d5.png)
            ![image] (https://github.com/FurMax/AndroidTest2/blob/image/d6.png)


    
    
    
 ### 二、模型训练：
 
 
 #### ① 导入相关的库
 


```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

#### ②： 模型训练

#### <1>:获取数据：

#### <2>:加载数据集：

    代码和运行结果如下：



```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    228813984/228813984 [==============================] - 93s 0us/step
    


```python
model = image_classifier.create(train_data)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-c6c91b75df2e> in <module>
    ----> 1 model = image_classifier.create(train_data)
    

    NameError: name 'train_data' is not defined



```python
model = image_classifier.create(train_data)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-c6c91b75df2e> in <module>
    ----> 1 model = image_classifier.create(train_data)
    

    NameError: name 'train_data' is not defined



```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    


```python
model = image_classifier.create(train_data)


```

    INFO:tensorflow:Retraining the models...
    


    ---------------------------------------------------------------------------

    TimeoutError                              Traceback (most recent call last)

    F:\SoftwareProgramPractice3\Anaconda3\lib\urllib\request.py in do_open(self, http_class, req, **http_conn_args)
       1353             try:
    -> 1354                 h.request(req.get_method(), req.selector, req.data, headers,
       1355                           encode_chunked=req.has_header('Transfer-encoding'))
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\http\client.py in request(self, method, url, body, headers, encode_chunked)
       1254         """Send a complete request to the server."""
    -> 1255         self._send_request(method, url, body, headers, encode_chunked)
       1256 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\http\client.py in _send_request(self, method, url, body, headers, encode_chunked)
       1300             body = _encode(body, 'body')
    -> 1301         self.endheaders(body, encode_chunked=encode_chunked)
       1302 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\http\client.py in endheaders(self, message_body, encode_chunked)
       1249             raise CannotSendHeader()
    -> 1250         self._send_output(message_body, encode_chunked=encode_chunked)
       1251 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\http\client.py in _send_output(self, message_body, encode_chunked)
       1009         del self._buffer[:]
    -> 1010         self.send(msg)
       1011 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\http\client.py in send(self, data)
        949             if self.auto_open:
    --> 950                 self.connect()
        951             else:
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\http\client.py in connect(self)
       1416 
    -> 1417             super().connect()
       1418 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\http\client.py in connect(self)
        920         """Connect to the host and port specified in __init__."""
    --> 921         self.sock = self._create_connection(
        922             (self.host,self.port), self.timeout, self.source_address)
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\socket.py in create_connection(address, timeout, source_address)
        807         try:
    --> 808             raise err
        809         finally:
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\socket.py in create_connection(address, timeout, source_address)
        795                 sock.bind(source_address)
    --> 796             sock.connect(sa)
        797             # Break explicitly a reference cycle
    

    TimeoutError: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。

    
    During handling of the above exception, another exception occurred:
    

    URLError                                  Traceback (most recent call last)

    <ipython-input-6-a08c9248a6d4> in <module>
    ----> 1 model = image_classifier.create(train_data)
          2 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_examples\lite\model_maker\core\task\image_classifier.py in create(cls, train_data, model_spec, validation_data, batch_size, epochs, steps_per_epoch, train_whole_model, dropout_rate, learning_rate, momentum, shuffle, use_augmentation, use_hub_library, warmup_steps, model_dir, do_train)
        337     if do_train:
        338       tf.compat.v1.logging.info('Retraining the models...')
    --> 339       image_classifier.train(train_data, validation_data, steps_per_epoch)
        340     else:
        341       # Used in evaluation.
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_examples\lite\model_maker\core\task\image_classifier.py in train(self, train_data, validation_data, hparams, steps_per_epoch)
        158       The tf.keras.callbacks.History object returned by tf.keras.Model.fit*().
        159     """
    --> 160     self.create_model()
        161     hparams = self._get_hparams_or_default(hparams)
        162 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_examples\lite\model_maker\core\task\image_classifier.py in create_model(self, hparams, with_loss_and_metrics)
        127     hparams = self._get_hparams_or_default(hparams)
        128 
    --> 129     module_layer = hub_loader.HubKerasLayerV1V2(
        130         self.model_spec.uri, trainable=hparams.do_fine_tuning)
        131     self.model = hub_lib.build_model(module_layer, hparams,
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\keras_layer.py in __init__(self, handle, trainable, arguments, _sentinel, tags, signature, signature_outputs_as_dict, output_key, output_shape, load_options, **kwargs)
        151 
        152     self._load_options = load_options
    --> 153     self._func = load_module(handle, tags, self._load_options)
        154     self._has_training_argument = func_has_training_argument(self._func)
        155     self._is_hub_module_v1 = getattr(self._func, "_is_hub_module_v1", False)
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\keras_layer.py in load_module(handle, tags, load_options)
        447       except ImportError:  # Expected before TF2.4.
        448         set_load_options = load_options
    --> 449     return module_v2.load(handle, tags=tags, options=set_load_options)
        450 
        451 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\module_v2.py in load(handle, tags, options)
         90   if not isinstance(handle, str):
         91     raise ValueError("Expected a string, got %s" % handle)
    ---> 92   module_path = resolve(handle)
         93   is_hub_module_v1 = tf.io.gfile.exists(
         94       native_module.get_module_proto_path(module_path))
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\module_v2.py in resolve(handle)
         45     A string representing the Module path.
         46   """
    ---> 47   return registry.resolver(handle)
         48 
         49 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\registry.py in __call__(self, *args, **kwargs)
         49     for impl in reversed(self._impls):
         50       if impl.is_supported(*args, **kwargs):
    ---> 51         return impl(*args, **kwargs)
         52       else:
         53         fails.append(type(impl).__name__)
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\compressed_module_resolver.py in __call__(self, handle)
         65           response, tmp_dir)
         66 
    ---> 67     return resolver.atomic_download(handle, download, module_dir,
         68                                     self._lock_file_timeout_sec())
         69 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\resolver.py in atomic_download(handle, download_fn, module_dir, lock_file_timeout_sec)
        416     logging.info("Downloading TF-Hub Module '%s'.", handle)
        417     tf.compat.v1.gfile.MakeDirs(tmp_dir)
    --> 418     download_fn(handle, tmp_dir)
        419     # Write module descriptor to capture information about which module was
        420     # downloaded by whom and when. The file stored at the same level as a
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\compressed_module_resolver.py in download(handle, tmp_dir)
         61       request = urllib.request.Request(
         62           self._append_compressed_format_query(handle))
    ---> 63       response = self._call_urlopen(request)
         64       return resolver.DownloadManager(handle).download_and_uncompress(
         65           response, tmp_dir)
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow_hub\resolver.py in _call_urlopen(self, request)
        520     # Overriding this method allows setting SSL context in Python 3.
        521     if self._context is None:
    --> 522       return urllib.request.urlopen(request)
        523     else:
        524       return urllib.request.urlopen(request, context=self._context)
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\urllib\request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        220     else:
        221         opener = _opener
    --> 222     return opener.open(url, data, timeout)
        223 
        224 def install_opener(opener):
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\urllib\request.py in open(self, fullurl, data, timeout)
        523 
        524         sys.audit('urllib.Request', req.full_url, req.data, req.headers, req.get_method())
    --> 525         response = self._open(req, data)
        526 
        527         # post-process response
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\urllib\request.py in _open(self, req, data)
        540 
        541         protocol = req.type
    --> 542         result = self._call_chain(self.handle_open, protocol, protocol +
        543                                   '_open', req)
        544         if result:
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\urllib\request.py in _call_chain(self, chain, kind, meth_name, *args)
        500         for handler in handlers:
        501             func = getattr(handler, meth_name)
    --> 502             result = func(*args)
        503             if result is not None:
        504                 return result
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\urllib\request.py in https_open(self, req)
       1395 
       1396         def https_open(self, req):
    -> 1397             return self.do_open(http.client.HTTPSConnection, req,
       1398                 context=self._context, check_hostname=self._check_hostname)
       1399 
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\urllib\request.py in do_open(self, http_class, req, **http_conn_args)
       1355                           encode_chunked=req.has_header('Transfer-encoding'))
       1356             except OSError as err: # timeout error
    -> 1357                 raise URLError(err)
       1358             r = h.getresponse()
       1359         except:
    

    URLError: <urlopen error [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。>



```python
inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')
inception_v3_spec.input_image_shape = [240, 240]
model = image_classifier.create(train_data, model_spec=inception_v3_spec)

```

    INFO:tensorflow:Retraining the models...
    

    INFO:tensorflow:Retraining the models...
    

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    

    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\keras\optimizers\optimizer_v2\gradient_descent.py:108: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    

    103/103 [==============================] - 60s 572ms/step - loss: 0.8928 - accuracy: 0.7558
    Epoch 2/5
    103/103 [==============================] - 62s 598ms/step - loss: 0.6618 - accuracy: 0.8911
    Epoch 3/5
    103/103 [==============================] - 61s 589ms/step - loss: 0.6240 - accuracy: 0.9154
    Epoch 4/5
    103/103 [==============================] - 61s 590ms/step - loss: 0.6043 - accuracy: 0.9275
    Epoch 5/5
    103/103 [==============================] - 59s 575ms/step - loss: 0.5931 - accuracy: 0.9360
    


#### 注意事项：

    1.下载模型过程中：使用默认模型会出现Timeout error 网络错误，将很难完成下载，故而本次实验使用在线模型下载，如以上代码和运行结果。
    
    2.除此之外，也可以将模型下载到本地：eg：inception_v3_spec = image_classifier.ModelSpec(uri='D:\Workspace\JupyterNotebookFiles\E5\efficientnet_lite0_feature-vector_2')
    
#### ③：评估模型：




```python
loss, accuracy = model.evaluate(test_data)

```

    12/12 [==============================] - 9s 543ms/step - loss: 0.6230 - accuracy: 0.9155
    


```python
model.export(export_dir='.')

```

    INFO:tensorflow:Assets written to: C:\Users\Fur\AppData\Local\Temp\tmpguynd0au\assets
    

    INFO:tensorflow:Assets written to: C:\Users\Fur\AppData\Local\Temp\tmpguynd0au\assets
    F:\SoftwareProgramPractice3\Anaconda3\lib\site-packages\tensorflow\lite\python\convert.py:766: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Saving labels in C:\Users\Fur\AppData\Local\Temp\tmpd4c9xvca\labels.txt
    

    INFO:tensorflow:Saving labels in C:\Users\Fur\AppData\Local\Temp\tmpd4c9xvca\labels.txt
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
    

    然后我们可以在notebook的工作目录下查看到模型文件：如下图：
        ![image] (https://github.com/FurMax/AndroidTest2/blob/image/r1.png)

