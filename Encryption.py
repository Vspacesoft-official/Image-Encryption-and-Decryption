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


"""# Reconstruct the original image from the scrambled image using the same key
reconstructed_img = np.bitwise_xor(scrambled_img, key)

# Display and save the reconstructed image
cv2.imshow('Reconstructed Image', reconstructed_img)
cv2.imwrite('reconstructed_image.png', reconstructed_img)
"""
cv2.waitKey(0)
cv2.destroyAllWindows()