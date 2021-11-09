from bertnet import BertNetModel, BertNet

model_ = BertNetModel()
model = BertNet(model_)

model.save('../local/test_save_model.pt', state_matrix_only=True)
model.load('../local/test_save_model.pt')