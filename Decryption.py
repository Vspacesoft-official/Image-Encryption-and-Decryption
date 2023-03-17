from PIL import Image

def decrypt_image(image_path, key):
    # Open the encrypted image
    encrypted_image = Image.open("1_encrypted_image.png")

    # Get the dimensions of the image
    width, height = encrypted_image.size

    # Create a new image to store the decrypted image
    decrypted_image = Image.new("RGB", (width, height))

    # Decrypt each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the encrypted pixel value
            encrypted_pixel = encrypted_image.getpixel((x, y))

            # Decrypt the pixel value
            decrypted_pixel = tuple((c * pow(key, -1, 256)) % 256 for c in encrypted_pixel)

            # Set the decrypted pixel value in the new image
            decrypted_image.putpixel((x, y), decrypted_pixel)

    # Save the decrypted image
    decrypted_image.save("1_decrypted_image.png")
    print("Image decrypted successfully.")


# Decrypt the image "encrypted_image.png" using key 13
decrypt_image("1_encrypted_image.png", 13)


# STEP 2

import cv2
import numpy as np

# Load the cover image and the embedded image
cover_image = cv2.imread("1_Result_satellite.png")
embedded_image = cv2.imread("1_decrypted_image.png")


# Define the alpha blending function
def alpha_blend(cover_image, embedded_image, alpha):
    # Rescale the embedded image to the same size as the cover image
    embedded_image = cv2.resize(embedded_image, (cover_image.shape[1], cover_image.shape[0]))

    # Convert the cover image and the embedded image to float32 data type
    cover_image = cover_image.astype(np.float32)
    embedded_image = embedded_image.astype(np.float32)

    # Compute the blended image
    blended_image = (1 - alpha) * cover_image + alpha * embedded_image

    # Convert the blended image back to uint8 data type
    blended_image = blended_image.astype(np.uint8)

    return blended_image


# Define the alpha blending factor
alpha = 0.000001

# Reverse the embedding using alpha blending
reverse_image = alpha_blend(cover_image, embedded_image, alpha)

# Save the reversed image
cv2.imwrite("1_reversed_image.png", reverse_image)

import cv2
import numpy as np

# Load the cover image and the embedded image
cover_image = cv2.imread("1_Result_scrambled_image.png")
embedded_image = cv2.imread("1_decrypted_image.png")


# Define the alpha blending function
def alpha_blend(cover_image, embedded_image, alpha):
    # Rescale the embedded image to the same size as the cover image
    embedded_image = cv2.resize(embedded_image, (cover_image.shape[1], cover_image.shape[0]))

    # Convert the cover image and the embedded image to float32 data type
    cover_image = cover_image.astype(np.float32)
    embedded_image = embedded_image.astype(np.float32)

    # Compute the blended image
    blended_image = (1 - alpha) * cover_image + alpha * embedded_image

    # Convert the blended image back to uint8 data type
    blended_image = blended_image.astype(np.uint8)

    return blended_image


# Define the alpha blending factor
alpha = 0.000001

# Reverse the embedding using alpha blending
reverse_image = alpha_blend(cover_image, embedded_image, alpha)

# Save the reversed image
cv2.imwrite("2_reversed_image.png", reverse_image)

# STEP 3

import cv2
import pywt

img = cv2.imread("1_reversed_image.png")

[LL1,(HL1,LH1,HH1)] = pywt.dwt2(img,'db2')
[LL2,(HL2,LH2,HH2)] = pywt.dwt2(LL1,'db2')
[LL3,(HL3,LH3,HH3)] = pywt.dwt2(LL2,'db2')

# Load the transformed image
LL3 = pywt.idwt2((LL3,(HL3,LH3,HH3)), 'db2')
LL2 = pywt.idwt2((LL2,(HL2,LH2,HH2)), 'db2')
LL1 = pywt.idwt2((LL1,(HL1,LH1,HH1)), 'db2')

# Convert the image back to BGR format and save the result
result = cv2.cvtColor(cv2.convertScaleAbs(LL1), cv2.COLOR_RGBA2BGR)
cv2.imwrite("1_reconstructed_cover_image.png", result)


import cv2
import pywt

img = cv2.imread("2_reversed_image.png")

[LL1,(HL1,LH1,HH1)] = pywt.dwt2(img,'db2')
[LL2,(HL2,LH2,HH2)] = pywt.dwt2(LL1,'db2')
[LL3,(HL3,LH3,HH3)] = pywt.dwt2(LL2,'db2')

# Load the transformed image
LL3 = pywt.idwt2((LL3,(HL3,LH3,HH3)), 'db2')
LL2 = pywt.idwt2((LL2,(HL2,LH2,HH2)), 'db2')
LL1 = pywt.idwt2((LL1,(HL1,LH1,HH1)), 'db2')

# Convert the image back to BGR format and save the result
result = cv2.cvtColor(cv2.convertScaleAbs(LL1), cv2.COLOR_RGBA2BGR)
cv2.imwrite("1_reconstructed_turtle_image.png", result)




