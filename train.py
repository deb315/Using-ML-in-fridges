from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="/content/drive/My Drive/project data sets/data_cleaned")
trainer.setTrainConfig(object_names_array=["brinjal","carrot","cauliflower","lemon","onion","potato","tomato"], batch_size=16, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()
