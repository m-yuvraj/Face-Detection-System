
# Face, Eye, & Gender Detection System

This repository contains a Python-based real-time face, eye, and gender detection system using **OpenCV** and **Tkinter**. The program captures video from your webcam, detects faces and eyes using Haar Cascade classifiers, and uses a pre-trained deep learning model to predict gender.

## Features
- **Real-time Face Detection**: The system detects faces from a webcam feed.
- **Eye Detection**: Identifies eyes within the detected faces.
- **Gender Prediction**: Classifies the gender (Male/Female) of the detected face using a pre-trained deep learning model.
- **User Interface**: The graphical interface is built using Tkinter, making it easy to interact with the detection system.
- **Multithreading**: The detection process runs in a separate thread, keeping the user interface responsive.

## Prerequisites
Make sure you have **Python 3.7+** installed along with the following libraries:

- **OpenCV** (`opencv-python` and `opencv-python-headless`):
  ```bash
  pip install opencv-python opencv-python-headless
  ```

- **Pillow (PIL)** for handling images in Tkinter:
  ```bash
  pip install pillow
  ```

Additionally, you will need pre-trained models for gender detection, such as:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

These files can be downloaded from [OpenCV's Model Zoo](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector).

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/face-eye-gender-detection.git
   cd face-eye-gender-detection
   ```

2. **Download Pre-trained Models**:
   Place the necessary model files (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`) in the project folder.

3. **Run the Application**:
   ```bash
   python main.py
   ```

4. **Start Detection**:
   Once the GUI opens, click the **Start Detection** button to begin real-time face, eye, and gender detection.

## How It Works
1. **Face Detection**: A Haar Cascade classifier is used to detect faces in the webcam feed.
2. **Eye Detection**: For each detected face, the system looks for eyes using another Haar Cascade classifier.
3. **Gender Prediction**: A deep learning model is used to predict the gender of the detected face. The result (Male or Female) is displayed.

### Example Output

When the detection system is running, the webcam feed will display detected faces and eyes, with rectangles drawn around them. Gender is predicted for the detected face and shown on top of the face.

## Project Structure

```
├── main.py                        # Main application script
├── README.md                      # Project description and instructions
├── deploy.prototxt                 # Deep learning model configuration (download required)
├── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained model for face/gender detection (download required)
```

### main.py
This is the core file that:
- Opens a Tkinter window to display the webcam feed.
- Detects faces and eyes in real-time using OpenCV.
- Predicts gender using a pre-trained deep learning model.

### deploy.prototxt & res10_300x300_ssd_iter_140000.caffemodel
These are pre-trained models needed for gender prediction. You can download them from OpenCV’s model zoo and place them in the project folder.

## Known Issues
- **Gender prediction accuracy** may vary depending on the webcam quality and lighting conditions.
- Only the first detected face is used for gender prediction. If multiple faces are present, the first one in the frame will be analyzed.

## Future Enhancements
- Implement **age prediction** alongside gender detection.
- Allow detection of **multiple faces** with gender prediction for each one.
- Add a **snapshot feature** to capture the detected face.
- Improve **gender detection accuracy** with a better model or deeper neural network.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change. Ensure your changes pass existing tests.

---
