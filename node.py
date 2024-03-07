import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

class TanhNewtonImplicitLayer(nn.Module):
    def __init__(self, out_features, tol = 1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter
  
    def forward(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break

                # newton step
                J = torch.eye(z.shape[1])[None,:,:] - (1 / torch.cosh(z_linear)**2)[:,:,None]*self.linear.weight[None,:,:]
                z = z - torch.solve(g[:,:,None], J)[0][:,:,0]
                self.iterations += 1
    
        # reengage autograd and add the gradient hook
        z = torch.tanh(self.linear(z) + x)
        z.register_hook(lambda grad : torch.solve(grad[:,:,None], J.transpose(1,2))[0][:,:,0])
        return z


# partly borrowed from ECE57000 assignments
def train(model, optimizer, train_loader, epoch, iter_num):
    model.to(device)
    model.train()
    loss_list = []
    run_loss = 0.0
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets) 
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        if (i+1 ) % iter_num == 0:
            # print ("[ epoch : %d, batch : %5d] loss : %.3f" %  (epoch, i+1, run_loss / iter_num) )
            loss_list.append(run_loss/iter_num)
            run_loss = 0.0
    return loss_list
    
def test(model, test_dataset):
    y_true = test_dataset.targets
    y_pred = torch.zeros_like(y_true)
    loss_fn = nn.CrossEntropyLoss()
    run_loss = 0.0
    loss_list = []
    model.eval()
    for i in range(len(test_dataset)):
        output = model(mnist_test[i][0])
        run_loss += loss_fn(output, y_true[i].reshape(1))
        y_pred[i] = torch.argmax(output).item()
        
    loss_list.append(run_loss)
    accuracy = accuracy_score(y_true, y_pred)
    print("Validation accuracy:", accuracy)
    return accuracy, loss_list

# Loading dataset

mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)

# Training our model

model = nn.Sequential(nn.Flatten(),
                      nn.Linear(784, 100),
                      TanhNewtonImplicitLayer(100, max_iter=40),
                      nn.Linear(100, 10)
                      ).to(device)
opt = optim.SGD(model.parameters(), lr=1e-1)

EPOCH = 10
train_loss = []
test_loss = []
acc_list = []
for i in range(EPOCH):
    if i == 5:
        opt.param_groups[0]["lr"] = 1e-2
    loss_list = train(model, opt, train_loader, i+1, 60)
    train_loss.extend(loss_list)
    accuracy, loss_list = test(model, mnist_test, 200)
    test_loss.extend(loss_list)
    acc_list.append(accuracy)
    
# Printing # of parameters    
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")

# Plotting Loss curves
plt.plot(np.linspace(0, EPOCH, len(train_loss)), train_loss, color='red', label = 'training loss')
plt.plot(np.linspace(0, EPOCH, len(test_loss)), test_loss, color='green', label = 'validation loss')

plt.xlabel("epochs")
plt.ylabel("Loss")
plt.title("Loss curves")
plt.ylim(0,0.5)
plt.legend()
plt.show()

# Plotting accuracy curve
plt.plot(np.linspace(0, EPOCH, len(acc_list)), acc_list, color='blue', label = 'validation accuracy curve')
plt.title("Validation accuracy")
plt.xlabel("epochs")
plt.legend()
plt.show()

# Showing the results
plt.figure(figsize=(15,12))
preds = []
labels = mnist_test.targets
for i in range(12):
    plt.subplot(3,4,i+1)
    preds.append(torch.argsort(model(mnist_test[i][0]), descending=True))
    plt.axis('off')
    plt.title("Pred: " + str(torch.argmax(model(mnist_test[i][0])).item()) + ", Label: " + str(labels[i].item()))
    plt.imshow(mnist_test[i][0][0], cmap='gray')

plt.show()


# Calculating the confusion matrix and accuracy
pred = [torch.argmax(model(mnist_test[i][0])).item() for i in range(len(mnist_test))]
accuracy = accuracy_score(labels, pred)
print("Accuracy:", accuracy)
conf_matrix = confusion_matrix(labels, pred)

# Visualize confusion matrix
plt.figure(figsize=(12,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.linspace(0,9,10).astype(int), yticklabels=np.linspace(0,9,10).astype(int))
plt.xlabel('Predicted Label', fontsize = 12)
plt.ylabel('true Label', fontsize = 12)
plt.title('Accuracy: '+ str(accuracy) + ', Test Data Size: '+str(len(mnist_test)), fontsize=15)
plt.show()

