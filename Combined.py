import cv2
import numpy as np

# Load the image
img = cv2.imread('dyp.png')

# Get the shape of the image
h, w, c = img.shape

# Generate a random key
key = np.random.randint(0, 256, size=(h, w, c), dtype=np.uint8)

# Scramble the image using XOR encryption
scrambled_img = np.bitwise_xor(img, key)

# Display and save the scrambled image
cv2.imshow('Scrambled Image', scrambled_img)
cv2.imwrite('1_scrambled_image.png', scrambled_img)

 # Reconstruct the original image from the scrambled image using the same key
reconstructed_img = np.bitwise_xor(scrambled_img, key)

# Display and save the reconstructed image
cv2.imshow('Reconstructed Image', reconstructed_img)
cv2.imwrite('2_reconstructed_image.png', reconstructed_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# dwt filter
import cv2
import pywt

# 3 DWT FOR cover image

img = cv2.imread('satellite.png')
img = cv2.resize(img,dsize=(512,512))
Img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.imwrite("1_Result_satellite.png",Img)

[LL1,(HL1,LH1,HH1)] = pywt.dwt2(Img,'db2')
[LL2,(HL2,LH2,HH2)] = pywt.dwt2(LL1,'db2')
[LL3,(HL3,LH3,HH3)] = pywt.dwt2(LL2,'db2')

import cv2
import pywt
# dwt filter for logo

img = cv2.imread("1_scrambled_image.png")
img = cv2.resize(img,dsize=(312,312))
Img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.imwrite("1_Result_scrambled_image.png", Img)

[LL1,(HL1,LH1,HH1)] = pywt.dwt2(Img,'db2')
[LL2,(HL2,LH2,HH2)] = pywt.dwt2(LL1,'db2')
[LL3,(HL3,LH3,HH3)] = pywt.dwt2(LL2,'db2')


# Alpha blending

import cv2
import numpy as np

# Load the cover image and the watermark logo
cover = cv2.imread('1_Result_satellite.png').astype(np.float32)
logo = cv2.imread('1_Result_scrambled_image.png').astype(np.float32)

# Resize the watermark logo to fit inside the cover image
logo = cv2.resize(logo, (int(cover.shape[1]/4), int(cover.shape[0]/4)))

# Choose a position to embed the watermark logo in the cover image
x = 50
y = 50

# Create a copy of the cover image and overlay the watermark logo on it using alpha blending technique
stego = cover.copy()
alpha = 0.5  # set the alpha value between 0 and 1
stego[y:y+logo.shape[0], x:x+logo.shape[1]] = alpha * logo + (1 - alpha) * stego[y:y+logo.shape[0], x:x+logo.shape[1]]

# Convert the stego image to uint8
stego = stego.astype(np.uint8)

# Save the resulting stego image
cv2.imwrite('1_stego.png', stego)


# ECCRYPTION

from PIL import Image

def multiplicative_cipher_encrypt(image_path, key):
    # Load the image
    image = Image.open("1_stego.png")

    # Get the dimensions of the image
    width, height = image.size

    # Convert the image into a list of pixels
    pixels = list(image.getdata())

    # Encrypt each pixel using the Multiplicative Cipher
    encrypted_pixels = []
    for pixel in pixels:
        r, g, b = pixel
        r_encrypted = (r * key) % 256
        g_encrypted = (g * key) % 256
        b_encrypted = (b * key) % 256
        encrypted_pixels.append((r_encrypted, g_encrypted, b_encrypted))

    # Create a new image from the encrypted pixels and save it
    encrypted_image = Image.new(image.mode, (width, height))
    encrypted_image.putdata(encrypted_pixels)
    encrypted_image.save("1_encrypted_image.png")

    print("Image encrypted successfully!")


image_path = "1_stego.png"
key = 13
multiplicative_cipher_encrypt(image_path, key)


# DECRYPTION PROCESS STARTS

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




