import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

mlflow.set_experiment("Assignment3_Hager")

learning_rate = 0.001
epochs = 10
batch_size = 64

X = torch.randn(200,10)
y = (torch.sum(X,dim=1)>0).float().unsqueeze(1)

model = nn.Sequential(
    nn.Linear(10,16),
    nn.ReLU(),
    nn.Linear(16,1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

with mlflow.start_run():

    mlflow.log_param("learning_rate",learning_rate)
    mlflow.log_param("epochs",epochs)
    mlflow.log_param("batch_size",batch_size)

    mlflow.set_tag("student_id","202202011")

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs=model(X)

        loss=criterion(outputs,y)

        loss.backward()

        optimizer.step()

        preds=(outputs>0.5).float()
        accuracy=(preds==y).float().mean()

        print(epoch,loss.item(),accuracy.item())

        mlflow.log_metric("loss",loss.item(),step=epoch)
        mlflow.log_metric("accuracy",accuracy.item(),step=epoch)

    mlflow.pytorch.log_model(model,"model")