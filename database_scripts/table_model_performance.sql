CREATE TABLE `model_performance` (
  `id` int NOT NULL AUTO_INCREMENT,
  `model_name` varchar(500) NOT NULL,
  `split_number` int NOT NULL,
  `epoch` int NOT NULL,
  `train_loss` float NOT NULL,
  `val_loss` float NOT NULL,
  `train_accuracy` float NOT NULL,
  `val_accuracy` float NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=81 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
SELECT * FROM travel_agency.model_performance;