import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score

# Load dataset
datasets = load_dataset("ylecun/mnist")

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return {'pixel_values': image, 'label': torch.tensor(label, dtype=torch.long)}

# Create datasets
train_dataset = CustomDataset(datasets['train']['image'], datasets['train']['label'], transform=transform)
eval_dataset = CustomDataset(datasets['test']['image'], datasets['test']['label'], transform=transform)

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, pixel_values, labels=None):
        x = pixel_values.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits

# Initialize model
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SimpleModel().to(device)

# Define compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# Configure TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=1024,
    gradient_accumulation_steps=1,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-4,
    save_strategy="steps",
    save_steps=10,
    max_grad_norm=1.0,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    eval_steps=10,
    eval_strategy="steps"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
torch.save(model.state_dict(), 'model_weights.bin')