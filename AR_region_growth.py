import numpy as np
import astropy.io.fits as fits
import os
import re

data_item = r"something.fits"
filename = os.path.basename(data_item)
regex_pattern = r"(\d{8})"
match = re.search(regex_pattern, filename)
if match:
    date = match.group(1)
    formatted_dir_name = f"HMI{date[:8]}"
    save_dir = formatted_dir_name
else:
    print(f"Cannot find the date in the filename! {filename}")
    save_dir = "HMI"


def get_pic_from_fits(data_item):
    source = fits.open(data_item)
    i = 0
    while not np.any(source[i].data):
        i += 1
    pic = np.array(source[i].data)
    return pic


def apply_intensity_segmentation(image):
    threshold = 50
    binary_image = np.where(np.abs(image) >= threshold, 0, 255)
    return binary_image


from scipy.ndimage import binary_opening, generate_binary_structure


def apply_morphological_opening(binary_image, struct_size):
    # Convert the struct_size from Mm to pixels, assuming 1 pixel = 1 Mm
    struct_size_in_pixels = struct_size

    # Create a structuring element
    struct_element = np.ones((struct_size_in_pixels, struct_size_in_pixels), dtype=bool)

    inverted_image = 255 - binary_image

    # Apply the morphological opening operation
    opened_image = binary_opening(inverted_image, structure=struct_element)

    # Convert the pixel values to 0 and 255
    opened_image = np.where(opened_image == 1, 0, 255)

    return opened_image


from collections import deque


def region_growing(image, kernel_image, intensity_threshold):
    # Get the kernel pixels from the binary image
    kernel_pixels = np.argwhere(kernel_image == 0)
    visited = np.zeros(image.shape, dtype=bool)
    # Initialize the region and the queue with the kernel pixels
    region = set()
    queue = deque(kernel_pixels)

    # While the queue is not empty
    while queue:
        pixel = queue.popleft()
        # Add the pixel to the region and the current region
        region.add((pixel[0], pixel[1]))
        visited[pixel[0], pixel[1]] = True
        # Enqueue the neighbors of the pixel
        for neighbor in get_neighbors(image, pixel, intensity_threshold):
            if neighbor not in region and not visited[neighbor[0], neighbor[1]]:
                queue.append(neighbor)
            visited[neighbor[0], neighbor[1]] = True
    return region


def get_neighbors(image, pixel, intensity_threshold):
    # Return the neighbors of the pixel in the image
    x, y = pixel
    neighbors = [
        (nx, ny)
        for nx in range(x - 1, x + 2)
        for ny in range(y - 1, y + 2)
        if 0 <= nx < image.shape[0]
        and 0 <= ny < image.shape[1]
        and (nx, ny) != (x, y)
        and np.abs(image[(nx, ny)]) > intensity_threshold
    ]
    return neighbors


from scipy.ndimage import binary_closing, generate_binary_structure


def apply_morphological_closing(binary_image, struct_size):
    # Convert the struct_size from Mm to pixels, assuming 1 pixel = 1 Mm
    struct_size_in_pixels = struct_size

    # Create a structuring element
    struct_element = np.ones((struct_size_in_pixels, struct_size_in_pixels), dtype=bool)

    inverted_image = 255 - binary_image
    # Apply the morphological closing operation
    closed_image = binary_closing(inverted_image, structure=struct_element)

    # Map the pixel values from 0 and 1 to 0 and 255
    closed_image = np.where(closed_image == 1, 0, 255)

    return closed_image


import matplotlib.pyplot as plt

# Load the input image
input_image = get_pic_from_fits(data_item)

# Module 1: intensity thresholding segmentation
kernel_image = apply_intensity_segmentation(input_image)

# Module 2: morphological opening operation
opened_image = apply_morphological_opening(kernel_image, struct_size=10)

# Module 3: region growing
region = region_growing(input_image, opened_image, intensity_threshold=50)


# Module 4: morphological closing operation
def create_image_from_region(region, image_shape):
    # Create an empty image
    image = np.ones(image_shape, dtype=np.uint8)
    image = image * 255
    # Set the pixels in the region to 0
    for pixel in region:
        image[pixel[0], pixel[1]] = 0
    return image


# Create an image from the region
region_image = create_image_from_region(region, input_image.shape)

# Apply the morphological closing operation
closed_image = apply_morphological_closing(region_image, struct_size=10)


# Display the images
# plt.figure(figsize=(15, 15))
# plt.subplot(231), plt.imshow(input_image, cmap='gray'), plt.title('Input Image')
# plt.subplot(232), plt.imshow(kernel_image, cmap='gray'), plt.title('Module 1: Kernel Pixels')
# plt.subplot(233), plt.imshow(opened_image, cmap='gray'), plt.title('Module 2: Opened Image')
# plt.subplot(234), plt.imshow(region_image, cmap='gray'), plt.title('Module 3: Region Growing')
# plt.subplot(235), plt.imshow(closed_image, cmap='gray'), plt.title('Module 4: Closed Image')

# save the result
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
ax.set_aspect(1)
ax.imshow(input_image, cmap="Greys")
alpha_matrix = np.ones_like(closed_image, dtype=float)
alpha_matrix[closed_image == 255] = 0
closed_image[input_image >= 0] = 255
ax.imshow(closed_image, cmap="gray", alpha=alpha_matrix)
plt.gca().invert_xaxis()
plt.savefig(f"pictures/{save_dir}/region_growth.png", dpi=300)
plt.show()
