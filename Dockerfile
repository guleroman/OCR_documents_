FROM ubuntu:16.04

RUN apt-get update && yes | apt-get upgrade
RUN mkdir -p /tensorflow/models
RUN apt-get install -y git python3 python3-dev python3-pip
apt-get install python3-pyqt5
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow
RUN apt-get install -y protobuf-compiler python3-pil python3-lxml
RUN pip3 install jupyter
RUN pip3 install matplotlib
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
RUN cd tensorflow/models/research/object_detection; git clone https://github.com/guleroman/OCR_documents_.git
RUN mv tensorflow/models/research/object_detection/OCR_documents_/* tensorflow/models/research/object_detection

WORKDIR /tensorflow/models/research

RUN apt-get install -y wget unzip
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip 
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.



#RUN protoc object_detection/protos/*.proto --python_out=.
RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


RUN pip3 install Cython
RUN pip3 install contextlib2  
RUN pip3 install pandas 
RUN pip3 install pillow 
RUN pip3 install opencv-contrib-python

RUN apt-get install -y libsm6 libxext6 tesseract-ocr libtesseract-dev libleptonica-dev pkg-config tesseract-ocr-rus
RUN CPPFLAGS=-I/usr/local/include pip3 install tesserocr

RUN  wget -O /tensorflow/models/research/object_detection/snils_graph/frozen_inference_graph.pb https://github.com/guleroman/OCR_documents_/raw/master/snils_graph/frozen_inference_graph.pb

RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

WORKDIR /tensorflow/models/research/object_detection

EXPOSE 8888


CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/tensorflow/models/research/object_detection", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
