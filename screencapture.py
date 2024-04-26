from PIL import ImageGrab
screenshot = ImageGrab.grab(bbox=(100, 100, 500, 500))

# Save the screenshot to a file
screenshot.save("screenshot.png")

# Close the screenshot
screenshot.close()

