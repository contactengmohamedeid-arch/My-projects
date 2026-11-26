# Game Scene Classifier
A small CNN in PyTorch that tells which game a screenshot is from: Elden Ring, Red Dead Redemption 2 or Resident Evil.

- `Video to images(Data).py` builds the dataset by saving one frame per second from a gameplay video.
- `Game Scene classifier.ipynb`: loads images with `ImageFolder` (resized to 128x128), two Conv2d layers (16 and 32 filters) then a linear layer to 3 classes, trains, saves `game_classifier.pth`, and tests on a single image.

**Run:** `pip install torch torchvision pillow opencv-python`, build the dataset folders, update the paths in the notebook, run all cells. The dataset is not in the repo. No accuracy claimed.
