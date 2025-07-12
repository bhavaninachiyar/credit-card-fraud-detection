import cv2
import openai
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Initialize OpenAI API (replace 'your-api-key' with your actual API key)
openai.api_key = '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'

# Load a pre-trained MobileNetV2 model for image recognition
model = MobileNetV2(weights='imagenet')

# Function to handle image recognition using MobileNetV2
def recognize_image(image_path):
    try:
        # Load the image using Keras and resize it to 224x224 pixels (as required by MobileNetV2)
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict the image
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions

        # Format the result
        result = "\n".join([f"{label}: {prob*100:.2f}%" for _, label, prob in decoded_preds])
        return result

    except Exception as e:
        return f"Error processing the image: {str(e)}"

# Function to get the chatbot response using OpenAI
def get_chatbot_response(prompt):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Updated model
            prompt=prompt,
            temperature=0.7,
            max_tokens=150
        )
        return response['choices'][0]['text'].strip()

    except Exception as e:
        return f"Error: {str(e)}"

# Chatbot interaction
def chatbot():
    print("Hello! I'm an image recognition chatbot.")
    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # Check if the input is an image file
        if user_input.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Image recognition function
            print("Chatbot: Recognizing the image, please wait...")
            result = recognize_image(user_input)
            print(f"Chatbot (Image Recognition): \n{result}")

        else:
            # Text-based conversation
            print("Chatbot: Thinking...")
            response = get_chatbot_response(user_input)
            print(f"Chatbot: {response}")

# Start the chatbot
if __name__ == "__main__":
    chatbot()
