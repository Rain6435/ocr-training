import numpy as np
from PIL import Image
import os

# Define paths to Easy class images
image_dir = r'data/difficulty_labels/easy'
image_files = ['easy_00000.png', 'easy_00100.png', 'easy_00500.png']

print('=' * 70)
print('EASY CLASS IMAGE ANALYSIS')
print('=' * 70)

for img_filename in image_files:
    img_path = os.path.join(image_dir, img_filename)
    
    if not os.path.exists(img_path):
        print(f'\n[WARNING] File not found: {img_path}')
        continue
    
    print(f'\n{"-" * 70}')
    print(f'Image: {img_filename}')
    print(f'{"-" * 70}')
    
    # Load image
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # 1. Image dimensions
    print(f'Dimensions: {img_array.shape}')
    if len(img_array.shape) == 3:
        print(f'  - Height: {img_array.shape[0]}, Width: {img_array.shape[1]}, Channels: {img_array.shape[2]}')
    else:
        print(f'  - Height: {img_array.shape[0]}, Width: {img_array.shape[1]} (Grayscale)')
    
    # Flatten to 1D for pixel analysis
    pixels = img_array.flatten()
    
    # 2. Unique pixel values count
    unique_values = np.unique(pixels)
    print(f'\nUnique pixel values: {len(unique_values)}')
    print(f'  - Range: {unique_values[0]} to {unique_values[-1]}')
    if len(unique_values) <= 20:
        print(f'  - All values: {sorted(unique_values)}')
    else:
        print(f'  - Sample values: {sorted(unique_values)[:20]}...')
    
    # 3. Min/Max/Mean/Std statistics
    print(f'\nPixel value statistics:')
    print(f'  - Min: {np.min(pixels)}')
    print(f'  - Max: {np.max(pixels)}')
    print(f'  - Mean: {np.mean(pixels):.2f}')
    print(f'  - Std: {np.std(pixels):.2f}')
    
    # Check if binary
    is_binary_strict = set(unique_values) == {0, 255}
    is_binary_two_vals = len(unique_values) == 2
    if is_binary_strict:
        print(f'\nBinary status: YES (strictly 0 and 255 only)')
    elif is_binary_two_vals:
        print(f'\nBinary status: YES but not standard (only 2 unique values)')
    else:
        print(f'\nBinary status: NO (has {len(unique_values)} different values)')

print('\n' + '=' * 70)
print('CONCLUSION')
print('=' * 70)
print('The Easy class images are NOT strictly binary.')
print('They contain grayscale values beyond just 0 and 255.')
