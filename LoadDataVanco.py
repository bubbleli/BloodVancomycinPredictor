import numpy as np

class LoadData(object):
    def __init__(self, no_userstate_features, no_forward_features, no_forward_train_samples, no_forward_validation_samples, no_forward_test_samples, no_backward_features, no_backward_train_samples, no_backward_validation_samples, no_backward_test_samples):
        self.no_userstate_features = no_userstate_features

        self.no_forward_features = no_forward_features
        self.no_forward_train_samples = no_forward_train_samples
        self.no_forward_validation_samples = no_forward_validation_samples
        self.no_forward_test_samples = no_forward_test_samples

        self.no_backward_features = no_backward_features
        self.no_backward_train_samples = no_backward_train_samples
        self.no_backward_validation_samples = no_backward_validation_samples
        self.no_backward_test_samples = no_backward_test_samples

        self.Train_data, self.Validation_data, self.Test_data = self.getDummyData()

    # Generate dummy data for the model
    def getDummyData(self):
        Train_data, Validation_data, Test_data = {}, {}, {}
        ## 1. Forward Pass ##
        # 1.1 user state features
        train_forward_userstate_X = np.random.rand(self.no_forward_train_samples, self.no_userstate_features).tolist()
        validation_forward_userstate_X = np.random.rand(self.no_forward_validation_samples, self.no_userstate_features).tolist()
        test_forward_userstate_X = np.random.rand(self.no_forward_test_samples, self.no_userstate_features).tolist()

        # 1.2 forward features
        train_forward_X = np.random.rand(self.no_forward_train_samples, self.no_forward_features).tolist()
        validation_forward_X = np.random.rand(self.no_forward_validation_samples, self.no_forward_features).tolist()
        test_forward_X = np.random.rand(self.no_forward_test_samples, self.no_forward_features).tolist()

        # 1.3 forward labels
        train_forward_Y = (10 * np.random.rand(self.no_forward_train_samples)).tolist()
        validation_forward_Y = (10 * np.random.rand(self.no_forward_validation_samples)).tolist()
        test_forward_Y = (10 * np.random.rand(self.no_forward_test_samples)).tolist()

        ## 2. Backward Pass ##
        # 2.1 user state features
        train_backward_userstate_X = np.random.rand(self.no_backward_train_samples, self.no_userstate_features).tolist()
        validation_backward_userstate_X = np.random.rand(self.no_backward_validation_samples, self.no_userstate_features).tolist()
        test_backward_userstate_X = np.random.rand(self.no_backward_test_samples, self.no_userstate_features).tolist()

        # 2.2 Backward features
        train_backward_X = np.random.rand(self.no_backward_train_samples, self.no_backward_features).tolist()
        validation_backward_X = np.random.rand(self.no_backward_validation_samples, self.no_backward_features).tolist()
        test_backward_X = np.random.rand(self.no_backward_test_samples, self.no_backward_features).tolist()

        # 2.3 Backward labels
        train_backward_Y = (10 * np.random.rand(self.no_backward_train_samples)).tolist()
        validation_backward_Y = (10 * np.random.rand(self.no_backward_validation_samples)).tolist()
        test_backward_Y = (10 * np.random.rand(self.no_backward_test_samples)).tolist()

        Train_data['FWD_User_X'] = train_forward_userstate_X        # data to determine user state at t
        Train_data['FWD_X'] = train_forward_X                       # other non-patient specific data needed for the forward prediction. E.g. dosage, time gap from t to t+1
        Train_data['FWD_Y'] = train_forward_Y                       # forward prediction label - vancomycin level at t+1
        Train_data['BWD_User_X'] = train_backward_userstate_X       # data to determine user state at t+1
        Train_data['BWD_X'] = train_backward_X                      # other non-patient specific data needed for the backward prediction. E.g. time gap
        Train_data['BWD_Y'] = train_backward_Y                      # backward prediction label - Dosage given at t

        Validation_data['FWD_User_X'] = validation_forward_userstate_X
        Validation_data['FWD_X'] = validation_forward_X
        Validation_data['FWD_Y'] = validation_forward_Y
        Validation_data['BWD_User_X'] = validation_backward_userstate_X
        Validation_data['BWD_X'] = validation_backward_X
        Validation_data['BWD_Y'] = validation_backward_Y

        Test_data['FWD_User_X'] = test_forward_userstate_X
        Test_data['FWD_X'] = test_forward_X
        Test_data['FWD_Y'] = test_forward_Y
        Test_data['BWD_User_X'] = test_backward_userstate_X
        Test_data['BWD_X'] = test_backward_X
        Test_data['BWD_Y'] = test_backward_Y

        return Train_data, Validation_data, Test_data
