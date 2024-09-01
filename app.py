import json
import streamlit as st
from src.api import client
from src.config import SYSTEM_MESSAGE, PHI_VISION_MODELS
from src.ui_components import file_upload, header, advanced_settings
from src.utils import prepare_content_with_images, all_images, all_videos


def main():
    # Title section
    header()

    model_option = st.sidebar.selectbox(
        "Select Phi Vision Model",
        PHI_VISION_MODELS.keys(),
        index=0,
    )
    MODEL = PHI_VISION_MODELS[model_option]

    # Add input field for system prompt
    st.sidebar.header("System Prompt")
    system_prompt = st.sidebar.text_area(
        label="Modify the prompt here.", value=SYSTEM_MESSAGE["content"]
    )
    SYSTEM_MESSAGE["content"] = system_prompt

    # Sidebar section for file upload
    uploaded_files, file_objects = file_upload()

    # Advanced Settings
    response_format = advanced_settings()

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    if "messages" in st.session_state.keys() and len(st.session_state.messages) > 0:
        # add clear chat history button to sidebar
        st.sidebar.button(
            "Clear Chat History",
            on_click=lambda: st.session_state.messages.clear(),
            type="primary",
        )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Display chat message in chat message container
        with st.chat_message(message["role"]):
            content = message["content"]
            if isinstance(content, list):
                st.markdown(content[0]["text"])
                urls = [item["image_url"]["url"] for item in content[1:]]
                st.image(urls if len(urls) < 3 else urls[0], width=200)
            else:
                st.markdown(content)

    if prompt := st.chat_input("Ask something", key="prompt"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            content = (
                prepare_content_with_images(prompt, file_objects)
                if file_objects
                else prompt
            )
            if file_objects:
                if all_images(uploaded_files):
                    st.image(uploaded_files, width=200)

                elif all_videos(uploaded_files):
                    st.video(uploaded_files[0], autoplay=True)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": content})

        # Get response from the assistant
        with st.chat_message("assistant"):
            messages = [SYSTEM_MESSAGE, *st.session_state.messages]
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
                response_format=response_format,
            )
            response = st.write_stream(stream)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
