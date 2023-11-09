南華大學人工智慧期中報告 
主題:初學者的 TensorFlow 2.0 教程
組員：11128037楊哲睿
作業流程如下

1.首先將TensorFlow 導入程序 
import tensorflow as tf

2.加載MNIST數據集。將樣本數據從整數轉換為浮點數：
mnist = tf.keras.datasets.mnist
  x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step

3.通過堆疊層来構建 tf.keras.Sequential 模型:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

4.對於每個樣本，模型都會返回一个包含 logits 或 log-odds，分數的向量，每個類一個:
predictions = model(x_train[:1]).numpy()
predictions
array([[ 0.0940896 , -0.07805067, -0.05265348,  0.42978847,  0.40682903,
        -0.09560589, -0.05340281, -0.19078785,  0.7025797 ,  0.05961688]],
      dtype=float32)

5.使用 losses.SparseCategoricalCrossentropy為訓練定義損失函數，它會接受 logits 向量和 True 索引，並為每个樣本返回一个標量損失:
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

6.此損失等於 true 類的負對數概率：如果模型确定類正确，。這個未經訓練的模型给出的概率接近隨機（每個類為 1/10），因此初始損失應該接近 -tf.math.log(1/10) ~= 2.3: 
loss_fn(y_train[:1], predictions).numpy()
2.5615656

7.在開始訓練之前，使用 Keras Model.compile 配置和編譯模型。 將 optimizer 類別設為 adam，將 loss 設定為您先前定義的 loss_fn 函數，並透過將 metrics 參數設為 accuracy 來指定要為模型評估的指標。
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

8.使用 Model.fit 方法調整您的模型參數並最小化損失：
model.fit(x_train, y_train, epochs=5)
Epoch 1/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0664 - accuracy: 0.9793
Epoch 2/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0588 - accuracy: 0.9809
Epoch 3/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0543 - accuracy: 0.9825
Epoch 4/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0477 - accuracy: 0.9840
Epoch 5/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0452 - accuracy: 0.9851
<keras.src.callbacks.History at 0x7e03ec53b340>

9.Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上檢查模型效能。
model.evaluate(x_test,  y_test, verbose=2)
313/313 - 1s - loss: 0.0745 - accuracy: 0.9795 - 652ms/epoch - 2ms/step
[0.07448458671569824, 0.9794999957084656]

10.現在，這個照片分類器的準確度已經接近 98%。 如果您想讓模型返回機率，可以封裝經過訓練的模型，並將 softmax 附加到該模型：
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[7.6116976e-08, 2.4037041e-11, 7.1538193e-07, 1.5976128e-05,
        1.7643511e-13, 1.9375362e-08, 6.5993830e-16, 9.9997187e-01,
        1.1468487e-09, 1.1308186e-05],
       [1.1328353e-13, 1.9851286e-05, 9.9998009e-01, 2.4689000e-09,
        2.4963244e-18, 2.0242071e-08, 8.8478969e-13, 9.3974197e-16,
        2.3560001e-11, 3.8124579e-17],
       [7.2058964e-10, 9.9968123e-01, 1.7641271e-06, 2.9356084e-07,
        7.0534298e-07, 7.9882668e-08, 2.6273523e-07, 2.9080219e-04,
        2.4886071e-05, 6.3338854e-09],
       [9.9998641e-01, 3.2977738e-13, 1.0213642e-05, 5.6859193e-08,
        1.6662804e-10, 2.1962487e-07, 3.2627722e-07, 1.6923141e-06,
        6.3021801e-09, 1.1109522e-06],
       [1.7757438e-06, 2.5145378e-12, 8.7661061e-09, 4.3601525e-10,
        9.9694037e-01, 4.0801208e-08, 1.4426221e-06, 3.1552644e-04,
        5.8961955e-09, 2.7409636e-03]], dtype=float32)> 
參考資料: https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh_cn



