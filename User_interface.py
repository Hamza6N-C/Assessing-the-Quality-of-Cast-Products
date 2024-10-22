import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from ultralytics import YOLO
import cv2

class ImageApp:
    def __init__(self, root):
        # Initialize the main application window
        self.root = root
        self.root.title("Image Detection App")  # Set the window title
        self.root.geometry("720x720")  # Set the window size
        ctk.set_appearance_mode("light")  # Set the appearance mode to light
        # ctk.set_default_color_theme("blue")  # Option to set the color theme

        # Create the main frame for the app with a transparent background
        self.main_frame = ctk.CTkFrame(root, fg_color="transparent", width=900, height=900)
        self.main_frame.pack(pady=40, padx=40, expand=True)  # Add padding and make the frame expandable

        # Frame for image display
        self.frame = self.main_frame

        self.root.update()  # Update the window to get correct dimensions
        self.set_background_image("webpage_background.jpg")  # Set the background image

        # Create the left frame with specified colors and dimensions
        self.left_frame = ctk.CTkFrame(self.frame, fg_color=("white", "gray20"), width=400, height=400)
        self.left_frame.pack(side="left", padx=10, expand=True)  # Add padding and make it expandable

        # Create the right frame with specified colors and dimensions
        self.right_frame = ctk.CTkFrame(self.frame, fg_color=("white", "gray20"), width=400, height=400)
        self.right_frame.pack(side="right", padx=10, expand=True)  # Add padding and make it expandable

        # Left frame contents
        self.image_label_left = ctk.CTkLabel(self.left_frame)  # Label for displaying the image
        self.image_label_left.pack(pady=(10, 5))  # Add vertical padding
        self.original_image = cv2.imread("add_image_icon.png")  # Load a default image
        self.display_image(self.original_image, self.image_label_left)  # Display the default image
        self.add_img_btn = ctk.CTkButton(self.left_frame, text="Add Image", command=self.add_image)  # Button to add an image
        self.add_img_btn.pack(pady=(5, 10))  # Add vertical padding

        # Right frame contents
        self.image_label_right = ctk.CTkLabel(self.right_frame, text="")  # Label for displaying results
        self.image_label_right.pack(pady=(10, 5))  # Add vertical padding
        self.original_image = cv2.imread("Result.png")  # Load a default result image
        self.display_image(self.original_image, self.image_label_right)  # Display the default result image
        self.show_results_btn = ctk.CTkButton(self.right_frame, text="Show Results", command=self.show_results)  # Button to show results
        self.show_results_btn.pack(pady=(5, 10))  # Add vertical padding

    def add_image(self):
        # Function to add an image
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])  # Open file dialog to select an image

        if self.image_path:
            self.original_image = cv2.imread(self.image_path)  # Read the selected image
            self.display_image(self.original_image, self.image_label_left)  # Display the selected image

    def show_results(self):
        # Function to show detection results
        if self.original_image is not None:
            model = YOLO("best.pt")  # Load the YOLO model
            results = model(self.original_image)  # Run the model on the image
            class_names = model.names  # Get class names

            r = results[0]  # Get the first result
            detected_object = int(r.probs.top1)  # Get the top detected object
            detected_object_name = class_names[detected_object]  # Get the name of the detected object
            detected_object_confidence_rate = r.probs.data[detected_object].item()  # Get the confidence rate
            detection_info = f"{detected_object_name}: {detected_object_confidence_rate:.4f}"  # Format detection info

            self.modified_image = self.original_image.copy()  # Copy the original image
            h, w, _ = self.modified_image.shape  # Get image dimensions
            text_position = (w // 2, h // 8)  # Set text position
            text_color = (0, 0, 255)  # Set text color (red)
            text_font = cv2.FONT_HERSHEY_SIMPLEX  # Set font type
            text_scale = 1  # Set text scale
            text_thickness = 2  # Set text thickness
            cv2.putText(self.modified_image, detection_info, text_position, text_font, text_scale, text_color, text_thickness)  # Add text to the image

            self.display_image(self.modified_image, self.image_label_right)  # Display the modified image

    def display_image(self, img, label):
        # Function to display an image in a given label
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
        img = Image.fromarray(img)  # Convert the image to a PIL Image
        img = img.resize((300, 300), Image.Resampling.LANCZOS)  # Resize the image
        ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(300, 300))  # Create a CTkImage
        label.configure(image=ctk_image)  # Set the image to the label
        label.image = ctk_image  # Keep a reference to the image

    def set_background_image(self, image_path):
        # Function to set the background image
        bg_image = Image.open(image_path)  # Open the background image
        bg_image = bg_image.resize((self.root.winfo_width(), self.root.winfo_height()), Image.Resampling.LANCZOS)  # Resize the image
        bg_photo = ctk.CTkImage(light_image=bg_image, dark_image=bg_image, size=(self.root.winfo_width(), self.root.winfo_height()))  # Create a CTkImage

        bg_label = ctk.CTkLabel(self.root, image=bg_photo, text="")  # Create a label for the background image
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Place the label to cover the entire window

        # Bring the main_frame to the front
        self.main_frame.lift()  # Lift the main frame above the background image

if __name__ == "__main__":
    root = ctk.CTk()  # Create the main application window
    app = ImageApp(root)  # Create an instance of the ImageApp class
    root.mainloop()  # Start the application main loop
