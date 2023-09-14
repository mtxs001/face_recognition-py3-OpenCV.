"""
作者：漫天星沙


cnn训练模型

"""
import random
import tensorflow as tf
import os
import face_data_predeal_002
from sklearn.model_selection import train_test_split

class CNN(tf.keras.Model):
    def __init__(self,user_num=2):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=[3, 3],padding='same', activation = tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3],activation=tf.nn.relu)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3],padding='same', activation=tf.nn.relu)
        self.conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3],activation=tf.nn.relu)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])
        self.flaten1 = tf.keras.layers.Flatten()
        # 扁平层，将除第一维度(也就是batch_size这一维)外的所有维度，都转成一个维度
        self.dense3 = tf.keras.layers.Dense(units=512,activation=tf.nn.relu)  # 全连接层
        self.dense4 = tf.keras.layers.Dense(units=user_num)  # 最后分类 5个单位
        self.user_num=user_num


    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool4(x)
        x = self.flaten1(x)
        x = self.dense3(x)
        x = self.dense4(x)
        output = tf.nn.softmax(x)
        return output

        # 识别人脸
    def face_predict(self, image):
        image = face_data_predeal_002.resize_image(image)
        image = image.reshape((1, face_data_predeal_002.IMAGE_SIZE, face_data_predeal_002.IMAGE_SIZE, 1))
        # 浮点并归一化
        image = image.astype('float32')
        image /= 255
        # 给出输入属于各个类别的概率
        result = self.predict(image)
        # 返回类别预测结果
        return result[0]

class Dataset:
    def __init__(self, path_name):

        # 训练集
        self.train_images = None
        self.train_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None
        # 数据集加载路径
        self.path_name = path_name
        # 当前库采用的维度顺序
        self.input_shape = None
        self.nb_classes = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=face_data_predeal_002.IMAGE_SIZE, img_cols=face_data_predeal_002.IMAGE_SIZE,img_channels=1):  # 灰度图 所以通道数为1 5个类别所以分组数为5
        # 加载数据集到内存
        images, labels,user_num = face_data_predeal_002.load_dataset(self.path_name)
        self.user_num=user_num
        train_images, test_images, train_labels, test_labels =train_test_split(images, labels, test_size=0.3,random_state=random.randint(0,200))  # 将总数据按0.3比重随机分配给训练集和测试集
        train_images = train_images.reshape(train_images.shape[0],img_rows, img_cols,img_channels)# 由于TensorFlow需要通道数，我们上一步设置为灰度图，所以这里为1，否则彩色图为3
        test_images = test_images.reshape(test_images.shape[0],img_rows, img_cols, img_channels)
        self.input_shape = (img_rows, img_cols, img_channels)
        # 输出训练集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')
        print(test_labels)
        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        # 将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        test_images /= 255
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels


if __name__ == '__main__':
    learning_rate = 0.001 # 学习率
    batch = 32 # batch数
    EPOCHS = 20 # 学习轮数
    dataset = Dataset('F:\pythonzuoye\shixun//0712\data') # 数据都保存在这个文件夹下
    dataset.load()
    model = CNN() # 模型初始化
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) #选择优化器
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy() # 选择损失函数
    train_loss = tf.keras.metrics.Mean(name='train_loss') # 设置变量保存训练集的损失值
    train_accuracy =tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # 设置变量保存训练集的准确值
    test_loss = tf.keras.metrics.Mean(name='test_loss') # 设置变量保存测试集的损失值
    test_accuracy =tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # 设置变量保存测试集的准确值
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape: #自动求解模型中可训练的变量的梯度
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # 优化器更新数据
        #计算出所有可训练变量的梯度，然后进行下一步的更新。
        train_loss(loss) # 更新损失值
        train_accuracy(labels, predictions) # 更新准确值
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    for epoch in range(EPOCHS):
        #训练开展的次数，训练多少轮
        train_ds =tf.data.Dataset.from_tensor_slices((dataset.train_images,dataset.train_labels)).shuffle(300).batch(batch)
        test_ds = tf.data.Dataset.from_tensor_slices((dataset.test_images,dataset.test_labels)).shuffle(300).batch(batch)
        #from_tensor_slices 切分传入的 Tensor 的第一个维度，生成相应的dataset 。
        #将python的list和numpy数组转换成tensorflow的dataset ，才能被model.fit函数训练
        #将多个数据特征进行分割并一一对应组合产生新的配对好的数据
        for images, labels in train_ds:
            train_step(images, labels) #训练集
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels) #测试集
        template = 'Epoch {} \nTrain Loss:{:.2f},Train Accuracy:{:.2%}\nTest Loss :{:.2f},Test Accuracy :{:.2%}'
        print(template.format(epoch + 1, train_loss.result(),train_accuracy.result(), test_loss.result(),test_accuracy.result())) # 打印
        model.save_weights('.\\face2') # 保存权重模型 命名为face
