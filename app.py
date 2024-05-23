import json
import os
import re

import streamlit as st
from PIL import Image

from phiai import Phi3VisionAI

# Define Phi3 model client
client = Phi3VisionAI()


# Define path to store chat history file
CHAT_HISTORY_FILE = "messages.json"
IMAGE_DIR = "uploaded_images"

# CSS to center crop the image in a circle
circle_image_css = """
<style>
.center-cropped {
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 50%;
    width: 96px;
    height: 96px;
    object-fit: cover;
}
</style>
"""

# Inject CSS
st.markdown(circle_image_css, unsafe_allow_html=True)


def main():
    # Title section
    display_header()

    # Sidebar section for image upload
    uploaded_file, image = image_upload()

    # Load chat history from file
    messages = load_chat_history()
    if messages is None:
        messages = []

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = messages

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # skip if role is system
        if message["role"] == "system":
            continue
        # Display chat message in chat message container
        with st.chat_message(message["role"]):
            content = message["content"]
            if content:
                # remove image token from content using re, if present, 
                # image token eg: "<|image_1|>", "<|image_2|>", etc
                content = re.sub(r"<\|image_\d\|>", "", content)
            st.markdown(content)

    # React to user input
    if prompt := st.chat_input(
        "Ask something", disabled=uploaded_file is None, key="prompt"
    ):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            # Placeholder function to send message to Phi3 model API
            # time.sleep(2)
            response = client.chat(st.session_state.messages, image)

            # Display Phi3 response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add Phi3 response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Save chat history before closing the app
    # save_chat_history(st.session_state.messages)

    # Made with section
    # st.markdown("---")
    # st.markdown("Made with ❤️ by [Bhimraj Yadav](https://github.com/bhimrazy)")


def display_header():
    st.markdown(
        """
        <img src="https://azure.microsoft.com/en-us/blog/wp-content/uploads/2024/05/Azure_Blog_Isometric_Illustration-12_1260x708.jpg" class="center-cropped">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='text-align: center;'>Chat with Phi-3-vision-128k-instruct</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='text-align: center; margin-bottom:4'>"
        "Phi-3-vision is a lightweight, state-of-the-art 4.2 billion parameter multimodal model with language and vision capabilities, available with a 128k context length. <a href='https://huggingface.co/microsoft/Phi-3-vision-128k-instruct' target='_blank'>Read more</a>"
        "<br>"
        "<p>Made with ❤️ by <a href='https://github.com/bhimrazy' target='_blank'>Bhimraj Yadav</a></p>"
        "</div>",
        unsafe_allow_html=True,
    )


def image_upload():
    st.sidebar.header("Upload an Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    image = None
    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.sidebar.error("File size exceeds 5 MB. Please upload a smaller file.")
        else:
            image = Image.open(uploaded_file)
            st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
            # Save uploaded image
            # save_uploaded_image(uploaded_file)
    return uploaded_file, image


def load_chat_history():
    # Load chat history from file
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            messages = json.load(file)
            return messages

    return None


def save_chat_history(messages):
    # Save chat history to file before closing the app
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(messages, file)


def save_uploaded_image(uploaded_file):
    # Save uploaded image to directory
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    with open(os.path.join(IMAGE_DIR, uploaded_file.name), "wb") as file:
        file.write(uploaded_file.getvalue())


if __name__ == "__main__":
    main()
