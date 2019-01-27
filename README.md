
# OCR_documentsV2
Optical Character Recognition of documents

## Introduction


## Steps
### 1. ��������� �������� Tensorflow � ����������� ����� Anaconda
#### 1.1 �������� ����������� Tensorflow Object Detection API � github

   �������� ����� �� ����� D: � �������� �� "pythonOCR". ���� ������� ������� ����� ��������� ��� ����� TensorFlow Object Detection, � ����� ��������� ������, ��������� ������ ���������, ����� ������������ � ��� ���������, ��� ���������� ��� ����������� ������������ ��� ��������.

��������� ������ ����������� TensorFlow Object Detection API, ������������� �� ������ https://github.com/tensorflow/models � ����� D:\pythonOCR.

#### 1.2 ��������� Faster_RCNN_Resnet101_Kitti �� TensorFlow's model zoo


[Faster_RCNN_Resnet101_Kitti](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)
���������� �� ������ ����� faster_rcnn_resnet101_kitti_2018_01_28 � ���������� D:\pythonOCR\models\research\object_detection.

#### 1.3 ��������� ������ �����������, ������������� �� ���� ��������. 
���������� ����� � ������� ����������� ��� ��������� � ��������� ��� ���������� ��������������� � D:\pythonOCR\models\research\object_detection.

#### 1.4 ��������� ����������� ����� Anaconda


```
D:\> conda create -n pythonOCR pip python=3.5
```


```
D:\> activate pythonOCR
```


```
(pythonOCR) D:\> conda install tensorflow-gpu
```


```
(pythonOCR) D:\> pip install tensorflow-gpu
```


```
(pythonOCR) D:\> conda install -c anaconda protobuf
(pythonOCR) D:\> conda install -c simonflueckiger tesserocr
(pythonOCR) D:\> pip install pillow
(pythonOCR) D:\> pip install lxml
(pythonOCR) D:\> pip install Cython
(pythonOCR) D:\> pip install jupyter
(pythonOCR) D:\> pip install matplotlib
(pythonOCR) D:\> pip install pandas
(pythonOCR) D:\> pip install opencv-python
```

#### 1.5 ��������� ���������� PYTHONPATH


```
(pythonOCR) D:\> set PYTHONPATH=D:\pythonOCR\models;D:\pythonOCR\models\research;D:\pythonOCR\models\research\slim
```

#### 1.6 ���������� ������ Protobuf 

� ��������� ������ Anaconda ��������� � ������� \models\research, ���������� � �������� � ��������� ������ ��������� ������� � ������� ������� ����:


```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
```

�������, ��������� ��������� ������� �� �������� D:\pythonOCR\models\research:


```
(pythonOCR) D:\pythonOCR\models\research> python setup.py build
(pythonOCR) D:\pythonOCR\models\research> python setup.py install
```

### 2. ������ 

#### 2.1 Jupiter notebook
�� ������ ��������� ������������� ��������, �������� OCR_snils.ipynb � Jupyter notebook. 
� �������� \object_detection ������� ��������� �������:


```
(pythonOCR) D:\pythonOCR\models\research\object_detection> jupyter notebook OCR_snils.ipynb
```

#### 2.2 OCR_snils.py

�����, �� ������ ��������� ����������� ���� ��� ������������� ��������� �� ��������� ������. � �������� \object_detection ������� ��������� �������:


```
(pythonOCR) D:\pythonOCR\models\research\object_detection> python OCR_snils.py --image='snils_data/image10.jpg'
```