

###   实验三：开发一个图片分类的安卓应用

##### 一、下载或Git Clone下开源的GitHub图片分类项目：

- 打开Github的TFLClassify开源项目网站，链接为：https://github.com/hoitab/TFLClassify
- 点击Clone按钮的download Zip 选项 下载源码包：截图如下：
- ![download](https://github.com/FurMax/AndroidTest2/blob/image/download.png)

​	

##### 二、运行初始未更改代码的项目:

- 首先，打开Android Studio，点击Open 找到下载的文件夹地址，打开此项目：
- 项目架构如下所示：
- ![项目架构图](https://github.com/FurMax/AndroidTest2/blob/image/项目架构图.png)
- 等待Gradle组件将所需的各种包资源下载完成后，运行虚拟机，运行结果如下：
- ![初始运行图](https://github.com/FurMax/AndroidTest2/blob/image/初始运行图.png)
- ![初始运行图2](https://github.com/FurMax/AndroidTest2/blob/image/初始运行图2.png)
- 由以上截图可见：APP下方的Fake Lable 0~2 表示图片识别各种物品的可能性数据。

##### 	三、导入模型文件并且更改Start模块的MainActivity.kt的代码：

- 首先，点击File<<New<<Other<<TensorFlow Lite Model 导入模型，截图如下：

- ![导入0](https://github.com/FurMax/AndroidTest2/blob/image/导入0.png)

- 然后选择finish模块下的ml文件夹下的FlowerModel.tflite文件导入

- ![导入1](https://github.com/FurMax/AndroidTest2/blob/image/导入1.png)

- 待导入成功后，将自动打开导入成功的FlowerModel.tflite文件的纲要：截图如下：

- ![导入模型图](https://github.com/FurMax/AndroidTest2/blob/image/导入模型图.png)

- 接下来，我们打开View-Tool Window - Todo 选项，这将展示此项目的所有Todo项：截图如下：

- ![todo0](https://github.com/FurMax/AndroidTest2/blob/image/todo0.png)

  

- 然后我们找到start模块的MainActivity.kt 的todo1，并且添加如下代码：

 ![todo1](https://github.com/FurMax/AndroidTest2/blob/image/todo1.png)

- todo2，添加如下代码：
- ![todo2](Chttps://github.com/FurMax/AndroidTest2/blob/image/todo2.png)
- todo3，添加如下代码：
- ![todo3](https://github.com/FurMax/AndroidTest2/blob/image/todo3.png)
- todo4，添加如下代码：
- ![todo4](https://github.com/FurMax/AndroidTest2/blob/image/todo4.png)
- 最后，我们注释一段代码，如下：
- ![todoEnd](https://github.com/FurMax/AndroidTest2/blob/image/todoEnd.png)



##### 四、运行一下最终更改完成的项目：





![result](https://github.com/FurMax/AndroidTest2/blob/image/result.png)