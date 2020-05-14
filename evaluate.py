from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="/content/drive/My Drive/project data sets/data_cleaned")
trainer.evaluateModel(model_path="/content/drive/My Drive/project data sets/data_cleaned/models", json_path="/content/drive/My Drive/project data sets/data_cleaned/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
