"""Create a simple test image for classifier testing"""
from PIL import Image
import numpy as np

# Create a simple 150x150 test image with some patterns
img_array = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save("test_image.jpg")
print("Created test_image.jpg (150x150)")
