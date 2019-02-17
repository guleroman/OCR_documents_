# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widget.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QFileDialog

class Object_detection_image():
    def main(name):
        import numpy as np
        import tensorflow as tf
        from PIL import Image

        import cv2
        import tesserocr
        import pandas as pd
        import json
        from utils import label_map_util
        from utils import visualization_utils as vis_util

        PATH_TO_CKPT = 'snils_graph/frozen_inference_graph.pb' # Путь к обученной модели нейросети
        PATH_TO_LABELS = 'training_snils/labelmap.pbtxt'  # Путь к label-файлу
        NUM_CLASSES = 1

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Распознавание интересующих полей на документе       
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                sess.run(tf.global_variables_initializer())
                image = Image.open(name)
                (im_width, im_height) = image.size 
                image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
                to_pixel = np.array([im_height, im_width, im_height, im_width])
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
				  
                image2 = cv2.imread(name)
                vis_util.visualize_boxes_and_labels_on_image_array(
					image2,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					use_normalized_coordinates=True,
					line_thickness=3,
					min_score_thresh=0.80)				  
                
        # Переведем координаты распознанных блоков в табличный вид        
        j = int(num_detections[0]) # Число выявленных блоков
        table = pd.DataFrame()
        tab_1 = []
        tab_2 = []
        tab_3 = []
        tab_4 = []
        for i in range(0,j):
            tab_1 = tab_1 + [boxes.T[1][i][0]]
            tab_2 = tab_2 + [boxes.T[3][i][0]]
            tab_3 = tab_3 + [boxes.T[0][i][0]]
            tab_4 = tab_4 + [boxes.T[2][i][0]]

        table['y min'] = tab_3
        table['x min'] = tab_1
        table['y max'] = tab_4
        table['x max'] = tab_2

        # Отсортируем таблицу по максимальному значению y
        table = table.sort_values('y max')

        # Переведем относительные координаты в координатные пиксели
        table = table * to_pixel

        # Произведем нарезку изображения на интересующие текстовые блоки
        images_new = []
        for i in range(0,j):
            img = image.crop( (int(table['x min'].iloc[i]),int(table['y min'].iloc[i]),int(table['x max'].iloc[i]),int(table['y max'].iloc[i])) ) #дата
            img.save('cropp_'+str(i)+'.jpg')
            img = cv2.imread('cropp_'+str(i)+'.jpg')
            img = Image.fromarray(cv2.GaussianBlur(img,(3,3),0))
            images_new = images_new + [img]

        # Применяем модуль tesserocr для OCR каждого изображения текстового блока в отдельности
        text = []
        for img in images_new:
            tex = tesserocr.image_to_text(img, lang='rus')
            tex = tex.replace(',', '.').replace("\n", '').replace("’", '').replace("'", '').replace('"', '').replace("?", '').replace("‘", '')
            text.append(tex) 
        for i in range(len(text),8):
            text.append('')
        # Структурируем информацию в словаре    
        data = {
            "number": text[0],
            "surname": text[1],
            "name": text[2],
            "patronymic": text[3],
            "birthday": text[4],
            "birthplace": text[5],
            "gender": text[6],   
            "registration": text[7]   
        }
        
        rgb_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        ui.label_4.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QtGui.QImage.Format_RGB888)))
        stroka = ''
        for key, value in data.items():
            stroka = key + ' - ' + value +'\n' + stroka
        ui.textBrowser.setText(stroka)

class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(841, 492)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("D:/unnamed.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Widget.setWindowIcon(icon)
        Widget.setWindowOpacity(1.0)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Widget)
        self.verticalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem1)
        self.label_2 = QtWidgets.QLabel(Widget)
        self.label_2.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Georgia")
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.line_2 = QtWidgets.QFrame(Widget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.label_4 = QtWidgets.QLabel(Widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setAutoFillBackground(True)
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setText("")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        spacerItem2 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.line = QtWidgets.QFrame(Widget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_5 = QtWidgets.QLabel(Widget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.textBrowser = QtWidgets.QTextBrowser(Widget)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_3.addWidget(self.textBrowser)
        self.label = QtWidgets.QLabel(Widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setSizeIncrement(QtCore.QSize(0, 0))
        self.label.setBaseSize(QtCore.QSize(0, 0))
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setLineWidth(19)
        self.label.setMidLineWidth(25)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("D:/461px-Generalstaff-278x300.png"))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(Widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setAutoFillBackground(True)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3, 0, QtCore.Qt.AlignTop)
        self.toolButton = QtWidgets.QToolButton(Widget)
        self.toolButton.setObjectName("toolButton")
        self.horizontalLayout_2.addWidget(self.toolButton, 0, QtCore.Qt.AlignTop)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.pushButton = QtWidgets.QPushButton(Widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("D:/machine-learning-analytics-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon1)
        self.pushButton.setIconSize(QtCore.QSize(30, 20))
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_3.addWidget(self.pushButton, 0, QtCore.Qt.AlignRight)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem3)
        self.verticalLayout_3.setStretch(4, 6)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 10)
        self.horizontalLayout.setStretch(3, 2)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        
        self.toolButton.clicked.connect(self.openExplorer)
        self.pushButton.clicked.connect(self.runTensorflow)

        self.retranslateUi(Widget)
        self.label_3.windowIconTextChanged['QString'].connect(self.label_3.setText)
        QtCore.QMetaObject.connectSlotsByName(Widget)
        
        
        
    def runTensorflow(self):
        Object_detection_image.main(str(ui.label_3.text()))

      
    def openExplorer(self):   
        class App(QWidget):
            def __init__(self):
                super().__init__()
                self.title = 'Выбор изображения..'
                self.left = 10
                self.top = 10
                self.width = 640
                self.height = 480
                self.initUI()
             
            def initUI(self):
                self.setWindowTitle('Выбор изображения..')
                self.setGeometry(self.left, self.top, self.width, self.height)
                self.openFileNameDialog()

            def openFileNameDialog(self):
                options = QFileDialog.Options()
                options |= QFileDialog.DontUseNativeDialog
                fileName, _ = QFileDialog.getOpenFileName(self,"Выбор изображения..", "","All Files (*);;Изображения (*.jpg)", options=options)
                if fileName:
                    ui.label_3.setText(fileName)
        add = App()

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "ObjectDetection"))
        self.label_2.setText(_translate("Widget", " Программный модуль распознавания текста на документах"))
        self.label_5.setText(_translate("Widget", "Вывод:"))
        self.label_3.setText(_translate("Widget", "..."))
        self.toolButton.setText(_translate("Widget", "..."))
        self.pushButton.setText(_translate("Widget", "Запуск"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Widget = QtWidgets.QWidget()
    ui = Ui_Widget()
    ui.setupUi(Widget)
    Widget.show()
    sys.exit(app.exec_())
