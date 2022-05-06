### 实验二：开发一个安卓应用

本实验目的在于开发一个Android Kotlin 项目，并且实现基本的跳转并且生成随机数的功能。



------



##### 一、做前期准备：

- 创建Android Kotlin配置基本环境

- 配置虚拟机

- 运行一个基本的安卓项目

  

##### 二、更改第一个界面的布局及代码：

1. 更改textview_first 的相关布局代码：

   ``

   ```xml
   <TextView
       android:id="@+id/textview_first"
       android:layout_width="96dp"
       android:layout_height="91dp"
       android:background="#00BCD4"
       android:text="@string/hello_first_fragment"
       android:textColor="#FAF9F9"
       android:textSize="72sp"
       android:textStyle="bold"
       app:layout_constraintBottom_toBottomOf="parent"
       app:layout_constraintEnd_toEndOf="parent"
       app:layout_constraintHorizontal_bias="0.6"
       app:layout_constraintStart_toStartOf="parent"
       app:layout_constraintTop_toTopOf="parent" />
       
   ```

2.更改按钮toast的相关布局代码：



```xml
<Button
    android:id="@+id/toast_button"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:layout_marginStart="24dp"
    android:layout_marginLeft="24dp"
    android:background="#2196F3"
    android:text="@string/button1"
    app:layout_constraintBottom_toBottomOf="parent"
    app:layout_constraintStart_toStartOf="parent"
    app:layout_constraintTop_toBottomOf="@+id/textview_first" />
```

3.更改按钮Random的相关布局代码：

```xml
<Button
    android:id="@+id/random_button"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:layout_marginEnd="24dp"
    android:layout_marginRight="24dp"
    android:background="#2196F3"
    android:text="Random"
    app:layout_constraintBottom_toBottomOf="parent"
    app:layout_constraintEnd_toEndOf="parent"
    app:layout_constraintTop_toBottomOf="@+id/textview_first" />
```

4.更改按钮Count的相关布局代码：

``

```xml
<Button
    android:id="@+id/count_button"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:background="#2196F3"
    android:text="@string/button"
    app:layout_constraintBottom_toBottomOf="parent"
    app:layout_constraintEnd_toStartOf="@+id/random_button"
    app:layout_constraintStart_toEndOf="@+id/toast_button"
    app:layout_constraintTop_toBottomOf="@+id/textview_first"
    />
```

5.给count按钮添加一个能够更新屏幕中央的数字的实现功能：

``

```kotlin
//        给count按钮添加一个click action
        view.findViewById<Button>(R.id.count_button).setOnClickListener {
            countMe(view)
        }
```

6.在CountMe方法内添加代码如下：

``

```kotlin
private fun countMe(view: View) {
    // Get the text view
    val showCountTextView = view.findViewById<TextView>(R.id.textview_first)

    // Get the value of the text view.
    val countString = showCountTextView.text.toString()

    // Convert value to a number and increment it
    var count = countString.toInt()
    count++

    // Display the new value in the text view.
    showCountTextView.text = count.toString()
}
```

7. 相关的String.xml文件代码如下:

   ``

   ```xml
   <resources>
       <string name="app_name">AndroidApplicationTest1</string>
       <string name="action_settings">Settings</string>
       <!-- Strings used for fragments for navigation -->
       <string name="first_fragment_label">First Fragment</string>
       <string name="second_fragment_label">Second Fragment</string>
       <string name="random_button_text">Random</string>
       <string name="previous">Previous</string>
   
       <string name="hello_first_fragment">0</string>
       <string name="hello_second_fragment">Hello second fragment. Arg: %1$s</string>
       <string name="button1">Toast</string>
       <string name="button">Count</string>
       <string name="random_heading">Here is a random number between 0 and %d.</string>
       <string name="textview1">R</string>
   </resources>
   ```

8.第一个界面的运行结果如下：

![result1-1](https://github.com/FurMax/AndroidTest2/blob/image/result1-1.png)

![result1-2](https://github.com/FurMax/AndroidTest2/blob/image/result1-2.png)

##### 三、完成第二个界面的开发：

1. 更新第二个界面 fragment_second.xml中的所有textview代码

   ``

   ```xml
   <TextView
       android:id="@+id/textview_header"
       android:layout_width="match_parent"
       android:layout_height="wrap_content"
       android:layout_marginStart="24dp"
       android:layout_marginLeft="24dp"
       android:layout_marginTop="24dp"
       android:layout_marginEnd="24dp"
       android:layout_marginRight="24dp"
       android:text="@string/random_heading"
       android:textColor="@color/colorPrimaryDark"
       android:textSize="24sp"
   
       app:layout_constraintEnd_toEndOf="parent"
       app:layout_constraintStart_toStartOf="parent"
       app:layout_constraintTop_toTopOf="parent" />
   ```

 ``

```xml
<TextView
    android:id="@+id/textview_random"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="@string/textview1"
    android:textColor="#FFFFFF"
    android:textSize="72sp"
    android:textStyle="bold"
    app:layout_constraintBottom_toTopOf="@+id/button_second"
    app:layout_constraintEnd_toEndOf="parent"
    app:layout_constraintStart_toStartOf="parent"
    app:layout_constraintTop_toBottomOf="@+id/textview_header"
    app:layout_constraintVertical_bias="0.45" />
```



2.更新button按钮的布局代码（也可以在design一步步设置）：

``

```xml
<Button
    android:id="@+id/button_second"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:layout_marginBottom="24dp"
    android:background="#00BCD4"
    android:text="@string/previous"
    app:layout_constraintBottom_toBottomOf="parent"
    app:layout_constraintEnd_toEndOf="parent"
    app:layout_constraintHorizontal_bias="0.498"
    app:layout_constraintStart_toStartOf="parent" />
```

3.更改界面的一些设置代码：

```xml
android:layout_width="match_parent"
android:layout_height="match_parent"
android:background="@color/screenBackground2"
tools:context=".SecondFragment">
```

4.检查导航图nav_graph:

![nav](https://github.com/FurMax/AndroidTest2/blob/image/nav.png)

   本项目选择Android的Basic Activity类型进行创建，默认情况下自带两个Fragments，并使用Android的导航机制Navigation。导航将使用按钮在两个Fragment之间进行跳转，就第一个Fragment修改后的Random按钮和第二个Fragment的Previous按钮。


5.启用SaleArgs组件：

​	SafeArgs 是一个 gradle 插件，它可以帮助您在导航图中输入需要传递的数据信息，作用类似于Activity之间传递数据的Bundle。

 	我们在bulid.gradle（Project）新增以下代码：

``

```properties
dependencies {
    def nav_version = "2.3.0-alpha02"
    classpath "androidx.navigation:navigation-safe-args-gradle-plugin:$nav_version"
    classpath "com.android.tools.build:gradle:4.2.0"
    classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version"

    // NOTE: Do not place your application dependencies here; they belong
    // in the individual module build.gradle files
}
```

​	在build.gradle(Module)新增以下代码：

```properties
plugins {
    id 'com.android.application'
    id 'kotlin-android'
    id 'androidx.navigation.safeargs'


}
```

​	和：

​	``

```properties
dependencies {

    implementation "org.jetbrains.kotlin:kotlin-stdlib:$kotlin_version"
    implementation 'androidx.core:core-ktx:1.3.1'
    implementation 'androidx.appcompat:appcompat:1.2.0'
    implementation 'com.google.android.material:material:1.2.1'
    implementation 'androidx.constraintlayout:constraintlayout:2.0.1'
    implementation 'androidx.navigation:navigation-fragment-ktx:2.3.0'
    implementation 'androidx.navigation:navigation-ui-ktx:2.3.0'
    testImplementation 'junit:junit:4.+'
    androidTestImplementation 'androidx.test.ext:junit:1.1.2'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.3.0'
}
```

6.第二个界面的运行结果如下：

![result2-1](https://github.com/FurMax/AndroidTest2/blob/image/result2-1.png)





至此，一个基于Android和Kotlin语言的项目已经完成。