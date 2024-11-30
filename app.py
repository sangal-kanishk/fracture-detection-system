from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the folder for saving uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the FractureModel class (matching your training setup)
class FractureModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int = 8):
        super(FractureModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=hidden_units*3, out_channels=hidden_units*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*4, out_channels=hidden_units*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*4*7*7, out_features=120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(0.3),
            nn.Linear(in_features=120, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv3(self.conv2(self.conv1(x))))

# Initialize the model
model = FractureModel(input_shape=3, output_shape=1, hidden_units=8)

# Load the model's state_dict
model.load_state_dict(torch.load('model/fracture_model_state_dict.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Update image transformation to remove normalization (to match training setup)
custom_image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # No normalization
])

@app.route('/')
def upload_image():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and transform the image
        img = Image.open(file).convert("RGB")
        img = custom_image_transform(img)
        img = img.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = model(img)
            pred_prob = torch.sigmoid(output).item()

        # Adjusted threshold for sensitivity
        if pred_prob > 0.0005:  # Adjust threshold if necessary
            result = "Fracture detected!"
        else:
            result = "No fracture detected."

        # Render the result page with the image and prediction
        return render_template('result.html', result=result, probability=pred_prob, image_filename=filename)

# Local Machine
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)

# if __name__ == "__main__":
#     app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
