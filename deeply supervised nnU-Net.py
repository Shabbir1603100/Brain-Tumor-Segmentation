import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CustomEncoder(nn.Module):
    def __init__(self, resnet, vgg):
        super(CustomEncoder, self).__init__()
        self.resnet = resnet
        self.vgg = vgg
        self.conv1 = nn.Conv3d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(p=0.5)
        
    def forward(self, x):
        x_resnet = self.resnet(x)
        x_vgg = self.vgg(x)
        x = torch.cat((x_resnet, x_vgg), dim=1)  # Concatenate along the channel dimension
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        return x

class CustomDecoder(nn.Module):
    def __init__(self):
        super(CustomDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.deconv5 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout3d(p=0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4 * 4, 1024)  # Adjust input dimensions accordingly
        self.fc2 = nn.Linear(1024, 3 * 64 * 64 * 64)  # Adjust output dimensions accordingly
        self.final_conv = nn.Conv3d(32, 3, kernel_size=1)  # Final segmentation layer
        
    def forward(self, x, encoder_features):
        x = self.deconv1(x)
        x = torch.cat((x, encoder_features[4]), dim=1)
        x = F.relu(self.conv1(x))
        
        x = self.deconv2(x)
        x = torch.cat((x, encoder_features[3]), dim=1)
        x = F.relu(self.conv2(x))
        
        x = self.deconv3(x)
        x = torch.cat((x, encoder_features[2]), dim=1)
        x = F.relu(self.conv3(x))
        
        x = self.deconv4(x)
        x = torch.cat((x, encoder_features[1]), dim=1)
        x = F.relu(self.conv4(x))
        
        x = self.deconv5(x)
        x = torch.cat((x, encoder_features[0]), dim=1)
        x = F.relu(self.conv5(x))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 32, 64, 64, 64)
        x = self.final_conv(x)
        
        return x

class DeepSupervisedUNet(nn.Module):
    def __init__(self, encoder):
        super(DeepSupervisedUNet, self).__init__()
        self.encoder = encoder
        self.decoder = CustomDecoder()
        
    def forward(self, x):
        encoder_features = []
        x_resnet = self.encoder.resnet(x)
        x_vgg = self.encoder.vgg(x)
        x = torch.cat((x_resnet, x_vgg), dim=1)  # Concatenate along the channel dimension
        
        x = F.relu(self.encoder.conv1(x))
        encoder_features.append(x)
        x = self.encoder.pool(x)
        x = self.encoder.dropout(x)
        
        x = F.relu(self.encoder.conv2(x))
        encoder_features.append(x)
        x = self.encoder.pool(x)
        x = self.encoder.dropout(x)
        
        x = F.relu(self.encoder.conv3(x))
        encoder_features.append(x)
        x = self.encoder.pool(x)
        x = self.encoder.dropout(x)
        
        x = F.relu(self.encoder.conv4(x))
        encoder_features.append(x)
        x = self.encoder.pool(x)
        x = self.encoder.dropout(x)
        
        x = F.relu(self.encoder.conv5(x))
        encoder_features.append(x)
        x = self.encoder.pool(x)
        x = self.encoder.dropout(x)
        
        x = self.decoder(x, encoder_features[::-1])  # Reverse the order of encoder features for decoding
        
        return x

# Define the ResNet34 and VGG16 encoders without pretrained weights
resnet34 = models.resnet34(pretrained=False)
vgg16 = models.vgg16(pretrained=False)

# Remove the final fully connected layers
resnet34 = nn.Sequential(*list(resnet34.children())[:-2])
vgg16 = nn.Sequential(*list(vgg16.children())[:-2])

# Create the encoder with the ensemble of ResNet34 and VGG16
encoder = CustomEncoder(resnet=resnet34, vgg=vgg16)

# Create the deeply supervised nnU-Net model
model = DeepSupervisedUNet(encoder=encoder)

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch['image'].unsqueeze(1).float()  # Adding channel dimension
            labels = batch['mask'].long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model
model = train_model(model, train_loader, criterion, optimizer)

# Function to evaluate the model predictions
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].unsqueeze(1).float()  # Adding channel dimension
            targets = batch['mask'].long()
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics for each region
    all_metrics = {'dice_whole': [], 'hd95_whole': [], 'dice_core': [], 'hd95_core': [], 'dice_enhance': [], 'hd95_enhance': []}
    
    for pred, target in zip(all_preds, all_targets):
        pred = np.argmax(pred, axis=0)
        metrics = evaluate_metrics(pred, target)
        for key, value in metrics.items():
            all_metrics[key].append(value)
    
    # Average the metrics over the dataset
    final_metrics = {key: np.nanmean(value) for key, value in all_metrics.items()}
    
    return final_metrics

# Evaluate the model on the validation set
model_metrics = evaluate_model(model, val_loader)
print(f'Model Metrics: {model_metrics}')
