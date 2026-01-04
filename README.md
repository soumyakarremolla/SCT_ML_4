# SCT_ML_4

**Hand Gesture Recognition (LeapGestRecog)** 🔧

Project that demonstrates loading and preprocessing the LeapGestRecog hand gesture dataset and training a small convolutional neural network (CNN) to classify hand gestures.

---

## ✅ Features

- Loads grayscale images organized by subject and gesture class
- Resizes images to a fixed resolution (`IMG_SIZE` = 64)
- Normalizes pixel values and reshapes data for CNN input
- Splits data using a stratified train/test split
- Trains a basic CNN using TensorFlow / Keras

## 📦 Requirements

Install dependencies from:`requirements.txt` (tested on Linux):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 📁 Dataset

This project expects the LeapGestRecog dataset to be placed at:

```
hand_gesture_recognition/data/leapGestRecog
```

The script expects the dataset to be arranged as:

```
hand_gesture_recognition/data/leapGestRecog/<subject>/<gesture>/*.png
```

> 💡 If you don't have the dataset, download it (LeapGestRecog) and extract it into the path above.

## ▶️ Usage

Run the main training script:

```bash
python main.py
```

This will:
- Load and preprocess images
- Build a small CNN
- Train for 5 epochs (default in `main.py`)
- Print test accuracy after training

You can change hyperparameters directly in `main.py` (e.g., `IMG_SIZE`, `epochs`, `batch_size`) or extend the script to accept CLI arguments.

## 📝 Notes

- Images are loaded in grayscale and expanded to a single-channel input for the CNN.
- The dataset labels are derived from folder names; ensure consistent naming across subjects.
- Current model and training settings are minimal — intended as a starting point for experimentation.

## Contributing

Contributions and improvements are welcome. Open a pull request or an issue with suggestions.

## License

This project is provided as-is (no formal license specified). Add a LICENSE file if you want to choose one.
