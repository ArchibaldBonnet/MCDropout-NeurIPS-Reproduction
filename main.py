import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

class MCDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.net(x)

def mc_dropout_predict(model, X_test, T=100):
    model.train()
    
    predictions = []
    with torch.no_grad(): 
        for _ in range(T):
            predictions.append(model(X_test).numpy())
            
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    variance = predictions.var(axis=0)
    
    return mean, variance

X_train = np.random.uniform(-2, 2, size=(200, 1)).astype(np.float32)
y_train = np.sin(3 * X_train) + np.random.normal(0, 0.1, size=X_train.shape).astype(np.float32)

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)

model = MCDropoutNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) 
criterion = nn.MSELoss()

print("Training")
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Training done")

X_test = np.linspace(-4, 4, 400).reshape(-1, 1).astype(np.float32)
X_test_tensor = torch.from_numpy(X_test)

mean, variance = mc_dropout_predict(model, X_test_tensor, T=100)
std = np.sqrt(variance)

plt.figure(figsize=(10, 5))

plt.scatter(X_train, y_train, color='red', s=10, label="Données d'entraînement")

plt.plot(X_test, mean, color='blue', label="Prédiction Moyenne")

for i in range(1, 4):
    plt.fill_between(X_test.flatten(), 
                     (mean - i * std).flatten(), 
                     (mean + i * std).flatten(), 
                     color='blue', alpha=0.1)

plt.axvline(x=2, linestyle='--', color='gray')
plt.axvline(x=-2, linestyle='--', color='gray')

plt.title("MC Dropout uncertainty")
plt.legend()
plt.show()