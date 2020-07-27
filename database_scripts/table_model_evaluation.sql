CREATE TABLE `model_evaluation` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `session_id` int NOT NULL,
  `precision` float NOT NULL,
  `recall` float NOT NULL,
  `mrr` float NOT NULL,
  `ndcg` float NOT NULL,
  `predictor_name` varchar(255) NOT NULL,
  `trivial_prediction` tinyint NOT NULL,
  `catalog_prediction` tinyint NOT NULL,
  `catalog_count` int NOT NULL,
  `ground_truth` int NOT NULL,
  `sequence` varchar(1500) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `input_sequence` varchar(1500) NOT NULL,
  `input_sequence_length` int NOT NULL,
  `user_sequence_length` int NOT NULL,
  `predictions` varchar(1500) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=245820 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
SELECT * FROM travel_agency.model_evaluation;