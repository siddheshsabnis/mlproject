def initiate_data_transformation(self, train_path, test_path):
    """
    Applies the preprocessing pipeline to the train and test datasets.
    """
    try:
        # Load the datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info("Train and test data loaded successfully")

        # Dynamically find the target column (if any)
        target_column_name = None
        for column in train_df.columns:
            if 'target' in column.lower():
                target_column_name = column
                break
        
        if target_column_name is None:
            raise CustomException("Target column not found in the dataset.", sys)

        input_features_train = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train = train_df[target_column_name]

        input_features_test = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test = test_df[target_column_name]

        # Get preprocessing object
        preprocessor = self.get_data_transformer_object()
        logging.info("Preprocessing object obtained")

        # Apply preprocessing
        input_features_train_transformed = preprocessor.fit_transform(input_features_train)
        input_features_test_transformed = preprocessor.transform(input_features_test)

        # Save the preprocessing object
        os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
        with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as file:
            pickle.dump(preprocessor, file)

        logging.info("Preprocessor saved to file")

        # Return transformed data
        return (
            np.c_[input_features_train_transformed, target_feature_train.to_numpy()],
            np.c_[input_features_test_transformed, target_feature_test.to_numpy()],
            self.data_transformation_config.preprocessor_obj_file_path
        )

    except Exception as e:
        raise CustomException(e, sys)
