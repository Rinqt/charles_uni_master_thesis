CREATE TABLE `lstm_models` (
  `id` int NOT NULL AUTO_INCREMENT,
  `model_type` varchar(255) NOT NULL,
  `training_type` varchar(450) NOT NULL,
  `description` varchar(3500) NOT NULL,
  `hyper_parameters` json NOT NULL,
  `meta_data` json NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=22 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='Table contains lstm models, their description and parameters';
SELECT * FROM travel_agency.lstm_models;