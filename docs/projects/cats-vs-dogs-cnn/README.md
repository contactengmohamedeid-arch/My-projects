# Cats vs. Dogs Image Classifier (CNN)
Convolutional neural network that classifies an image as a cat or a dog.

**File:** [`notebooks/Image Classification using CNN.ipynb`](../../../notebooks/Image%20Classification%20using%20CNN.ipynb)

**How it works**
- TensorFlow / Keras. Images are loaded from folders with `ImageDataGenerator` (augmentation) at 150x150, batch size 32, with separate train and validation generators.
- Three Conv2D + MaxPooling2D blocks (32, 64, 128 filters), then Dense layers of 128 and 64 units and a single output unit for the two-class decision.
- Trained for 15 epochs.

**Run:** `pip install tensorflow`, point the generators at your image folders, run all cells.
**Not in the repo:** the image dataset.
