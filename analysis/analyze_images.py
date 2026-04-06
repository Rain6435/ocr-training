import numpy as np
from PIL import Image
from pathlib import Path

# Find images
data_dir = Path('data/raw/iam')
all_images = sorted(list(data_dir.rglob('*.png')))[:6]

print(f"Analyzing {len(all_images)} sample images:\n")

for i, img_path in enumerate(all_images):
    img = Image.open(img_path)
    img_array = np.array(img)
    
    dimensions = img_array.shape
    unique_values = len(np.unique(img_array))
    min_val = img_array.min()
    max_val = img_array.max()
    mean_val = img_array.mean()
    std_val = img_array.std()
    
    print(f"Image {i+1}: {img_path.name}")
    print(f"  Dimensions: {dimensions}")
    print(f"  Unique pixel values: {unique_values}")
    print(f"  Min/Max: {min_val}/{max_val}")
    print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
    print()
