"""
作者：漫天星沙
"""

"""
打开摄像头，保存人脸到本地。
"""
import cv2
import os


def CreateFolder(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)


def face_pic_capture(window_name, camera_id, catch_pic_num, path_name):
    CreateFolder('data/'+path_name)  # CreateFolder()函数用于检测传递过来的文件路径是否存在，如果路径不存在就创建该路径。
    cv2.namedWindow(window_name)

    image_set = cv2.VideoCapture(camera_id)  #cv2.VideoCapture()是OpenCV的一个API，参数camera_id为0，表示打开笔记本的内置摄像头，如果是视频文件路径则打开本地视频，有的设为1表示打开外接摄像头。
    classifier = cv2.CascadeClassifier(r"F:\python\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")

    num = 1  # 用于对每张图片命名
    while image_set.isOpened():  #判断视频对象是否成功读取，成功读取视频对象返回True。
        flag, frame = image_set.read()  # 按帧读取视频，返回值flag是布尔型，正确读取则返回True，读取失败或读取视频结尾则会返回False；frame为每一帧的图像，为三维矩阵、BGR格式。
        if not flag:
            break
        image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRect = classifier.detectMultiScale(image_grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))  #OpenCV2中的人脸检测函数，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示），函数由分类器对象调用。

        if len(faceRect) > 0:  # 大于0则检测到人脸
            for facet in faceRect:  #对同一个画面有可能出现多张人脸，所以这里用一个for循环将所有检测到的人脸都读取出来，然后返回检测到的每张人脸在图像中的起始坐标（左上角，x、y）以及长、宽（h、w）。
                x, y, w, h = facet

                if w > 100:
                    img_name = 'data/{}/{}.jpg'.format(path_name, num)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    image = frame[y - 10: y + h + 10, x - 5: x + w + 5]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(img_name, image)  # 检测到有人脸的图像帧保存到本地
                    cv2.rectangle(frame, (x - 5, y - 10), (x + w + 5, y + h + 10), (0, 0, 255), 3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # 显示当前保存了多少张图片，
                    cv2.putText(frame, "num:{}".format(num), (x + 30, y - 15), font, 1, (0, 250, 250), 4)

                    num += 1
                    if num > catch_pic_num:
                        break

        if num > catch_pic_num:
            break

        # 显示图像,按"Q"键中断采集过程
        cv2.imshow(window_name, frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # 释放摄像头并关闭销毁所有窗口
    image_set.release()
    cv2.destroyAllWindows()


window_name='face_catch_screen'
window_name=window_name.encode("gbk").decode('UTF-8', errors='ignore')
camera_id=0
catch_pic_num=200
path_name=input('输入你的名字：')
face_pic_capture(window_name, camera_id, catch_pic_num, path_name)