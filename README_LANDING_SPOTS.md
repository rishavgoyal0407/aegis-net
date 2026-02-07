# Drone Landing Spot Detection System

A computer vision system that analyzes drone footage to identify safe landing spots and visualizes them using color-coded heatmaps.

## Features

- **Multi-Factor Safety Analysis**:
  - Terrain flatness (edge detection)
  - Surface texture uniformity
  - Gradient/slope analysis
  - Color uniformity detection

- **Heatmap Visualization**:
  - Red zones = Unsafe areas (low safety score)
  - Green zones = Safe areas (high safety score)
  - Real-time analysis in separate windows (original + heatmap)
  - Side-by-side comparison view

- **Flexible Processing**:
  - Analyze single images
  - Process video footage
  - Export analyzed videos
  - Generate detailed matplotlib heatmaps

## How It Works

The system divides each frame into a grid (default 20x20) and analyzes each cell using:

1. **Flatness Score (35% weight)**: Uses Canny edge detection - fewer edges indicate flatter surfaces
2. **Texture Score (25% weight)**: Analyzes pixel variance - lower variance means smoother surfaces
3. **Gradient Score (30% weight)**: Calculates surface slope using Sobel operators
4. **Color Uniformity (10% weight)**: Measures color consistency across the region

Each cell receives a safety score from 0-100, which is visualized in the heatmap.

## Usage

### Quick Test
```bash
python test_landing_spots.py
```
This will analyze the sample drone image and display results.

### Interactive Mode
```bash
python agent.py
```
Choose from:
1. Analyze drone video (with real-time visualization)
2. Analyze single image
3. Process video and save output

### Programmatic Usage

```python
from agent import LandingSpotDetector

# Create detector
detector = LandingSpotDetector(grid_size=20)

# Analyze an image
safety_map, heatmap = detector.process_image("photos/drone_img.jpg")

# Process video
detector.process_video("videos/drone_flight.mp4", sample_rate=15)

# Get raw analysis
import cv2
frame = cv2.imread("image.jpg")
safety_map, heatmap_overlay = detector.analyze_frame(frame)

# Find best spot
import numpy as np
best_spot = np.unravel_index(np.argmax(safety_map), safety_map.shape)
score = safety_map[best_spot]
print(f"Best landing spot at grid ({best_spot[1]}, {best_spot[0]}) with score {score:.2f}")
```

## Controls (Video Mode)

**Two windows will appear during video analysis:**
- **Original Drone Footage**: Shows the raw video feed
- **Landing Spot Heatmap**: Shows the safety analysis overlay with color-coded zones

**Keyboard controls:**
- **'q'**: Quit/exit
- **'s'**: Save current frame analysis (both original and heatmap)
- Any key: Close image windows

## Output Files

- `landing_heatmap_overlay.jpg`: Original image with heatmap overlay
- `safety_heatmap.png`: Detailed matplotlib heatmap with colorbar
- `landing_analysis.jpg`: Saved heatmap from video analysis (when 's' is pressed)
- `original_frame.jpg`: Saved original frame (when 's' is pressed during video)

## Customization

### Adjust Grid Resolution
```python
detector = LandingSpotDetector(grid_size=30)  # Finer analysis
detector = LandingSpotDetector(grid_size=10)  # Coarser/faster analysis
```

### Adjust Safety Score Weights
Edit the weights in `calculate_safety_score()` method:
```python
weights = {
    'flatness': 0.35,   # Importance of flat surfaces
    'texture': 0.25,    # Importance of smooth texture
    'gradient': 0.30,   # Importance of low slope
    'color': 0.10       # Importance of color uniformity
}
```

### Video Processing Speed
```python
# Process every Nth frame (higher = faster but less accurate)
detector.process_video("video.mp4", sample_rate=30)  # Every 30th frame
detector.process_video("video.mp4", sample_rate=5)   # Every 5th frame (slower)
```

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- Matplotlib

## Example Results

After running on drone footage, you'll see:
- **Safety scores**: 0-100 scale for each grid cell
- **Best landing spot**: Grid position and score
- **Visual heatmap**: Color-coded overlay showing safe (green) and unsafe (red) areas
- **Top ranking**: List of the safest landing locations

## Tips for Best Results

1. **Image Quality**: Higher resolution images provide more accurate analysis
2. **Grid Size**: Adjust based on your needs - larger grids are faster but less precise
3. **Sampling Rate**: For long videos, increase sample_rate to speed up processing
4. **Lighting**: Works best with consistent lighting conditions
5. **Camera Angle**: Overhead/top-down views work best for landing spot analysis

## Safety Disclaimer

This system is a computer vision tool designed to assist in landing spot identification. It should NOT be the sole decision-making tool for actual drone landings. Always:
- Use professional judgment
- Verify conditions in person when possible
- Consider factors beyond visual analysis (wind, terrain stability, obstacles)
- Follow all aviation regulations and safety protocols
