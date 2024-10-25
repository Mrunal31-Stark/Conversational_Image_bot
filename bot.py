import tkinter as tk
from tkinter import filedialog, Text, messagebox , ttk
from PIL import Image, ImageTk , ImageDraw
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

# Load processors and models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Helper functions
def generate_caption(image_path):
    """Generate caption for the uploaded image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cpu")
    outputs = caption_model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def answer_question(image_path, question):
    """Generate an answer based on the image and the user's question."""
    # Open the image again and process it for the question-answering step
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to("cpu")  # Adjust to "cuda" if GPU available
    outputs = qa_model.generate(**inputs)  # Pass both pixel_values and input_ids
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# GUI setup
def select_image():
    """Open file dialog to select an image and display it in the GUI."""
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        img = Image.open(image_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        caption_text.delete("1.0", tk.END)  # Clear previous text
        caption_text.insert(tk.END, "Image loaded. Click 'Generate Caption' to proceed.")

# Wrapper function for generating caption
def generate_image_caption():
    """Wrapper to call generate_caption with the image path."""
    if not image_path:
        messagebox.showerror("Error", "Please select an image first.")
        return
    caption = generate_caption(image_path)
    caption_text.delete("1.0", tk.END)
    caption_text.insert(tk.END, f"Caption: {caption}")
    global current_caption
    current_caption = caption

def ask_question():
    """Take user question, generate an answer, and display it."""
    question = question_entry.get()
    if not current_caption:
        messagebox.showerror("Error", "Please generate a caption first.")
        return
    if not question.strip():
        messagebox.showerror("Error", "Please enter a question.")
        return
    answer = answer_question(image_path, question)
    answer_text.delete("1.0", tk.END)
    answer_text.insert(tk.END, f"Answer: {answer}")
    
    

def create_rounded_button(parent, text, command, color, width=15):
    """Create a rounded button using PIL."""
    # Create an image with rounded edges
    img = Image.new("RGBA", (120, 40), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, 120, 40), radius=20, fill=color)
    img_tk = ImageTk.PhotoImage(img)
    
    # Create button with image
    button = ttk.Button(parent, text=text, command=command, image=img_tk, compound="center")
    button.image = img_tk  # Keep a reference to the image
    button.config(width=width)
    button.pack(pady=5)

# Tkinter window
root = tk.Tk()
root.title("Conversational Image Bot")
root.geometry("600x600")  # Set a fixed height for better layout
root.configure(bg="lightgray")  # Set a simple background color

# Image display section
image_label = tk.Label(root, bg="lightgray")
image_label.pack(pady=10)

# Select image button
create_rounded_button(root, "Select Image", select_image, "#FF6F61")

# Caption display
caption_text = Text(root, height=4, width=50, bg="#F0F8FF", font=("Arial", 12), wrap="word")
caption_text.pack(pady=5)

# Button to generate caption
create_rounded_button(root, "Generate Caption", generate_image_caption, "#FF7F50")

# Question entry
question_label = tk.Label(root, text="Ask a question about the image:", bg="lightgray")
question_label.pack(pady=5)

question_entry = tk.Entry(root, width=52, bg="white", borderwidth=1)
question_entry.pack(pady=5)

# Answer display
answer_text = Text(root, height=4, width=50, bg="#F0F8FF", font=("Arial", 12), wrap="word")
answer_text.pack(pady=5)

# Ask question button
create_rounded_button(root, "Get Answer", ask_question, "#20B2AA")

# Run GUI
current_caption = None
image_path = ""
root.mainloop()