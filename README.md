# Application of VLMs in Robotics
### Abstract
Vision Language Models (VLMs) have recently emerged as a powerful class of multimodal foundation models capable of jointly reasoning over visual inputs and natural language. Unlike traditional deep learning models in robotics, which are typically trained on task-specific small-scale datasets, foundation models are pretrained on internet-scale data, granting them superior generalization capabilities and, in many cases, emergent zero-shot reasoning abilities across unseen tasks.

Integrating the visual and linguistic capabilities of VLMs could significantly expand the applications of robotics, making robotic systems more versatile, intuitive, and capable of interacting with humans in a natural way. Such integration opens up possibilities across a wide range of domains, including natural human-robot interaction, enriched navigation, semantic mapping, descriptive navigation, multimodal perception, object detection, and scene understanding.

This work explores the application of VLMs to mobile robotics, with a focus on high-level embodied tasks such as Visual-Language Navigation (VLN) and Embodied Question Answering (EQA).

To connect VLMs with low-level robot functionality, we introduce a novel function-calling interface that allows the model to invoke perception and control tools, such as acquiring images, querying positions, or issuing motion commands, autonomously in response to natural language instructions.

We evaluate our approach in the AI2-Thor simulator, enabling rigorous testing in indoor environments. Our results demonstrate that pretrained VLMs, when properly interfaced through tool-augmented architectures, are capable of grounding language in action and perception for closed-loop control. These findings point toward the promising role of foundation models in advancing general-purpose language-driven robot autonomy.

## Getting Started
### Installation
Clone the github repository to your local directory:
```bash
git clone https://github.com/tommasoTubaldo/Application_of_VLMs_in_Robotics.git
```

Install all necessary dependencies:
```bash
pip install -r requirements.txt
pip install -q -U google-genai
```
> **Note**: *Python 3.9+* is required for using the google-genai API.

Set up the Gemini API key:
1) Go to https://aistudio.google.com/prompts/new_chat
2) Log-in or register with you google credentials
2) Click on **"Get API key"** on the top part
3) Click on **"Create API key"** and copy the API key
4) On the terminal:
   ```bash
   cd ~/your_project
   export GEMINI_API_KEY="<your_api_key>"
    ```
