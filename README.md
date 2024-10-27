# Laser Pointer Detection using OpenCV

This Python script detects a red laser pointer in real-time using OpenCV. It captures video from a webcam, processes each frame to identify the laser spot, and highlights it on the display.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)

## Installation

Install the required libraries:

```bash
pip install opencv-python numpy
```

## Details
In order to more accurately identify a red laser pointer and avoid misjudging some red objects as red laser pointers, the following techniques are used in the code to solve the problem:
- def adaptive_mean_filter( Remove noise while preserving image details and edges )
- brightness_threshold( Added brightness limit )
- cv2.cvtColor( Convert the image to HSV format to filter out red laser pointers from multiple perspectives )
- red_mask（ Enhance the detection of red regions in images by calculating the advantages of the red channel over the green and blue channels ）
- Apply erosion and dilation to remove small noise
- More details show in the code...

## Usage
Run the script:
```bash
python detect_laser_pointer.py
```

