<div align="center">
  <img src="https://github.com/user-attachments/assets/ed441961-912a-4db2-9043-3ba4c7cf0b0e" height="150"/>
  <br/>
  <h1>Chat with Phi 3.5/3 Vision LLMs</h1>
  <a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-and-chat-with-phi-3-vision-128k-instruct">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
  </a><br/>
<!--   <a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-and-chat-with-phi-3-vision-128k-instruct"></a> -->
<!--   <img src="https://github.com/user-attachments/assets/3cfab380-0fa6-4430-af21-ac5fff3928ee" alt="Chat with Phi 3.5/3 Vision LLMs" width="640" height="360"> -->
<video src="https://github.com/user-attachments/assets/9af93c91-7d27-48f5-8cee-5f7dbfb024a3" 
       type="video/mp4" 
       controls 
       style="max-width: 640px; width: 100%; height: auto;">
</video>
</div>

## Overview

[**Phi-3.5-vision**](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision.

This model enables multi-frame image understanding, image comparison, multi-image summarization/storytelling, and video summarization, which have broad applications in office scenarios.

## Getting Started

Follow these steps to set up and run the project:

### 1. Install Dependencies  

i. **Download and Install NVIDIA CUDA**  
   Visit the [NVIDIA CUDA Toolkit Downloads page](https://developer.nvidia.com/cuda-downloads) and follow the instructions to install CUDA compatible with your system.  

ii. **Install Required Python Packages**  
   Ensure you have all the necessary dependencies installed by running the following commands:  

   ```bash  
   pip install -r requirements.txt  
   pip install flash_attn  
   ```  

   If you encounter any issues while installing `flash_attn`, refer to the [FlashAttention Installation Guide](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) for troubleshooting tips and additional setup details.  

### 2. Start the API Server

Launch the API server powered by [LitServe](https://github.com/Lightning-AI/LitServe):

```bash
python server.py
```

### 3. Launch the Streamlit App

Start the Streamlit application with the following command:

```bash
streamlit run app.py
```

## About

This project is developed and maintained with ❤️ by [Bhimraj Yadav](https://github.com/bhimrazy).
