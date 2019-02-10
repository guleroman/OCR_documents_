# python OCR_snils.py --image=snils_data/image10.jpg

import numpy as np
import tensorflow as tf
from PIL import Image

import cv2
import tesserocr
import pandas as pd
import json

PATH_TO_CKPT = '/snils_graph/frozen_inference_graph.pb' # Путь к обученной модели нейросети
PATH_TO_LABELS = '/training_snils/labelmap.pbtxt'  # Путь к label-файлу
NUM_CLASSES = 1

flags = tf.app.flags
flags.DEFINE_string('image', '', 'Path to the image')
FLAGS = flags.FLAGS

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
        image = Image.open(FLAGS.image)
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
    img = image.crop( (table['x min'].iloc[i],table['y min'].iloc[i],table['x max'].iloc[i],table['y max'].iloc[i]) ) #дата
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
print (data)
# Производим запись в JSON файл
with open("data_file.json", "w") as write_file:
    json.dump(data, write_file)
	
print('Successfully! Create data_file.json')
