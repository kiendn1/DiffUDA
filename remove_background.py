from rembg import remove
from PIL import Image

# Paths to the input images and the output image
input_path_a = '/home/user/code/DiffUDA/Office Home/Art/Ruler/00002.jpg'
input_path_b = '/home/user/code/DiffUDA/Office Home/Art/Backpack/00001.jpg'
output_path = 'result.png'

# Load Image A and remove its background
image_a = Image.open(input_path_a)
image_a_no_bg = remove(image_a)

# Resize the extracted object from Image A
resize_factor = 0.55  # Adjust the resize factor as needed
new_size = (int(image_a_no_bg.width * resize_factor), int(image_a_no_bg.height * resize_factor))
resized_image_a_no_bg = image_a_no_bg.resize(new_size, Image.LANCZOS)

# Load Image B
image_b = Image.open(input_path_b)

# Calculate position to paste the resized object onto Image B
position = ((image_b.width - resized_image_a_no_bg.width) // 2,
            (image_b.height - resized_image_a_no_bg.height) // 2)

# Paste the resized object onto Image B
image_b.paste(resized_image_a_no_bg, position, resized_image_a_no_bg)

# Save the final image
image_b.save(output_path)
