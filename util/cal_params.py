import models
from torchsummary import summary


# 2S2F
summary(models.DynamicsEvolver(1, 4, 64, 2, 2, 0.0, True, 'cpu'), input_size=(1, 4), batch_size=-1)
summary(models.DynamicsEvolver(1, 4, 64, 2, 2, 0.0, False, 'cpu'), input_size=(1, 4), batch_size=-1)
summary(models.LSTM(in_channels=1, feature_dim=4), input_size=(1, 4), batch_size=-1)
summary(models.TCN(input_size=4, output_size=4, num_channels=[32,16,8], kernel_size=3, dropout=0.1), input_size=(1, 4), batch_size=-1)
summary(models.NeuralODE(in_channels=1, feature_dim=4), input_size=(1, 4), batch_size=-1)

# 1S2F
summary(models.DynamicsEvolver(1, 3, 64, 2, 2, 0.0, True, 'cpu'), input_size=(1, 3), batch_size=-1)
summary(models.DynamicsEvolver(1, 3, 64, 2, 2, 0.0, False, 'cpu'), input_size=(1, 3), batch_size=-1)
summary(models.LSTM(in_channels=1, feature_dim=3), input_size=(1, 3), batch_size=-1)
summary(models.TCN(input_size=3, output_size=3, num_channels=[32,16,8], kernel_size=3, dropout=0.1), input_size=(1, 3), batch_size=-1)
summary(models.NeuralODE(in_channels=1, feature_dim=3), input_size=(1, 3), batch_size=-1)