#encoding=utf-8
import network
import datetime
import mnist_loader
from PIL import Image

### loading data
training_data,_,test_data = mnist_loader.load_data_wrapper()
#
# print("training data")
# print(type(training_data))
# print(len(training_data))
# print(training_data[0][0].shape)
# print(training_data[0][1].shape)
# print(training_data[0])

#
# print("test data")
# print(len(test_data))



begin = datetime.datetime.now()
# 设置每层神经元个数，第一层和最后一层必须是784和10
net=network.Network([784,30,10],cost = network.CrossEntropyCost)
# 参数，第一个参数表示训练集，第二个参数开始分别表示 epcho, mini batch size，eta学习率，lmbda正则化参数
net.SGD(training_data,30,10,0.5,5.0,evaluation_data=test_data)
end = datetime.datetime.now()
print "训练耗时：",(end - begin)
print
print "最优训练集错误率为：%.4f"%net.low_training_error
print "最优测试集错误率为：%.4f"%net.low_test_error


'''
# 导入一张图片，输出预测结果
one_picture_data = Image.open(" ")
one_picture_input = np.reshape(one_picture_data, (784, 1))
print "this picture is %s"%net.feedforward(one_picture_input)
'''



