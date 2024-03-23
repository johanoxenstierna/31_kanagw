import cv2
import numpy as np

# Load the Kanagawa painting image
image = cv2.imread('./pictures/kanagawa_720.png')

# Define the water area (you may need to adjust this manually)
# This could involve manual selection or using image processing techniques
# For simplicity, let's assume water is in the lower part of the image
water_area = image.shape[0] // 2  # Adjust this based on your image

# Define parameters for the wave animation
amplitude = 10  # Amplitude of the wave
frequency = 0.1  # Frequency of the wave

# Create a video writer object to save the animation
height, width, _ = image.shape
out = cv2.VideoWriter('animated_kanagawa.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Animation loop
for t in range(100):  # Adjust the number of frames as needed
    # Generate a sine wave to simulate water motion
    wave = amplitude * np.sin(2 * np.pi * frequency * t)

    # Create a copy of the original image
    animated_image = np.copy(image)

    # Apply the wave effect to the water area
    animated_image[water_area:, :] = np.roll(animated_image[water_area:, :], int(wave), axis=0)

    # Write the frame to the video
    out.write(animated_image)

    # Display the animated image (optional)
    cv2.imshow('Animated Kanagawa', animated_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video writer and close the OpenCV windows
out.release()
cv2.destroyAllWindows()
