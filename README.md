# Fracture Detection System

[![Deployment Status](https://img.shields.io/badge/Deployment-Active-green)](https://web-production-e4a9.up.railway.app/)

A **Fracture Detection System** using deep learning to detect fractures from medical X-ray images. This project leverages the power of **PyTorch** and **Torchvision** to build and train a Convolutional Neural Network (CNN) model. The model is served via a **Flask** application, providing real-time predictions via a user-friendly web interface.

## 🌟 Features

- **Real-time fracture detection**: Upload an X-ray image, and the model will instantly predict whether a fracture is present.
- **Pre-trained CNN**: Built using PyTorch with transfer learning techniques for optimal performance.
- **Clean UI/UX**: Intuitive user interface designed for seamless interaction.
- **Deployed on Railway.app**: Stable, fast, and efficient deployment with zero downtime.

## 🚀 Live Demo

[Live on Railway.app](https://web-production-e4a9.up.railway.app/)

## 📸 Screenshots

- Upload page for X-ray images
![pic1](https://github.com/prabhmeharbedi/fracture-detection/blob/main/images/pic1.png?raw=true)

- Upload page for X-ray images
![pic2](https://github.com/prabhmeharbedi/fracture-detection/blob/main/images/pic2.png?raw=true)

- Real-time prediction with fracture classification
![pic3](https://github.com/prabhmeharbedi/fracture-detection/blob/main/images/pic3.png?raw=true)

- Real-time prediction with fracture classification
![pic4](https://github.com/prabhmeharbedi/fracture-detection/blob/main/images/pic4.png?raw=true)

## 🔧 Technologies Used

- **Backend**: Flask
- **Frontend**: HTML, CSS, Bootstrap
- **Machine Learning**: PyTorch, Torchvision
- **Deployment**: Railway.app

## 📂 Project Structure

```bash
   ├── app.py            # Flask    application
   ├── model             # Pre-trained CNN model files
   ├── templates         # HTML templates for the UI
   ├── static            # Static files (CSS, JS)
   ├── requirements.txt  # Project dependencies
   └── README.md         # Project documentation (this file)
```

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.9+**
- **Pip** (Python package installer)
- **Git**

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/prabhmeharbedi/fracture-detection.git
   cd fracture-detection

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For MacOS/Linux

3. **Install the Dependencies**:

   ```bash
   pip install -r requirements.txt

4. **Run the app locally**:

   ```bash
   python app.py

The app will be running on `http://127.0.0.1:5000/`.

## 🚀 Deployment

To deploy this project, follow the [Railway.app Deployment Guide](https://railway.app/docs/deployments). Here's a quick guide:

1. **Install the Railway CLI**: Follow the [installation guide](https://railway.app/docs/cli).
2. **Link your project**: Use the CLI to link your project using the `railway link` command.
3. **Deploy**: Run `railway.app` to deploy your project.

## 🧠 Model Architecture

The model is a **Convolutional Neural Network (CNN)** built with **PyTorch**. It contains multiple convolutional layers followed by fully connected layers that output a binary classification: **fracture** or **no fracture**.

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the project**.

2. **Create a new branch**:
   ```bash
   git checkout -b feature-branch

3. **Commit your changes**:
   ```bash
   git commit -m 'Add some feature'

4. **Push to the branch**:
   ```bash
   git push origin feature-branch

5. **Open a pull request**.


## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👤 Author

**Prabhmehar Bedi**

- GitHub: [prabhmeharbedi](https://github.com/prabhmeharbedi)
- LinkedIn: [Prabhmehar Bedi](https://www.linkedin.com/in/prabhmeharbedi/)
