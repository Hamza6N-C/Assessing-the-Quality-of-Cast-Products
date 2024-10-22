import os  # Module for interacting with the operating system (e.g., checking if folders exist, creating folders)
import sys  # Module for accessing accessing command-line arguments
import cv2  # OpenCV library for computer vision and image processing
import numpy as np  # NumPy library for numerical operations

def add_gaussian_noise(image, mean=0, sigma=60):
    """
    Add Gaussian noise to an image.
    """
    h, w, c = image.shape  # Get the height, width, and number of channels of the image
    gauss = np.random.normal(mean, sigma, (h, w, c))  # Generate Gaussian noise with the specified mean and sigma
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)  # Add the noise to the image and clip the values to the range [0, 255]
    return noisy_image  # Return the noisy image

def resize_image(image, scale_percent=50):
    """
    Resize an image.
    """
    width = int(image.shape[1] * scale_percent / 100)  # Calculate the new width based on the scale percentage
    height = int(image.shape[0] * scale_percent / 100)  # Calculate the new height based on the scale percentage
    dim = (width, height)  # Create a tuple with the new dimensions
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)  # Resize the image using OpenCV's resize function
    #ensures that the resized image looks smooth and natural, preserving details as much as possible
    return resized_image  # Return the resized image

def main():
    # Check if folder path is provided as command-line argument
    if len(sys.argv) != 2: #checks if exactly two arguments are provided when running the script
        print("Usage: python script.py <folder_path>")
        return
    #(sys.argv[0] is the script name itself, and sys.argv[1] should be the folder path).
    folder_path = sys.argv[1]  # Get the folder path from the command-line argument

    # checks if the directory specified by folder_path does not exist or is not a valid directory
    if not os.path.isdir(folder_path): 
        print("Folder path does not exist.")
        return

    # Create a new folder for noisy images
    noisy_folder_path = os.path.join(folder_path, "noised_images")
    os.makedirs(noisy_folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Load all JPEG images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)  # Load the image using OpenCV

        # Apply Gaussian noise
        noisy_image = add_gaussian_noise(image)

        # Resize the image
        resized_noisy_image = resize_image(noisy_image)
        resized_noisy_image = noisy_image  

        # Save the noisy image
        new_image_name = "n_" + image_file  # Prefix the filename with "n_"
        noisy_image_path = os.path.join(noisy_folder_path, new_image_name)
        cv2.imwrite(noisy_image_path, resized_noisy_image)  # Save the noisy image using OpenCV

        print(f"Noisy image saved: {noisy_image_path}")

    print("All images processed successfully.")

if __name__ == "__main__":
    main()  # Run the main function
