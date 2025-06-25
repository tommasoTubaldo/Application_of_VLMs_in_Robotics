# Application of VLMs in Robotics

This repository presents the implementation and experiments associated with our research on applying Vision-Language Models (VLMs) to mobile robotics. VLMs are a recent class of multimodal foundation models capable of joint reasoning over images and natural language, offering strong generalization and emergent zero-shot capabilities across diverse tasks.

### 🔎 What This Project Does
We connect the reasoning abilities of VLMs with low-level robotic control using a custom function-calling interface, enabling the model to autonomously:

- Capture and interpret visual inputs (e.g., camera images)

- Query the robot's state (e.g., position, orientation)

- Execute motion commands (e.g., move, rotate)

This setup empowers the robot to understand and act upon natural language instructions in real-time, enabling complex embodied tasks such as:

- Visual-Language Navigation (VLN)

- Embodied Question Answering (EQA)

- Semantic mapping and scene understanding

- Natural and multimodal human-robot interaction

### 🤔 Why It Matters
By leveraging foundation models, this project pushes toward general-purpose, intuitive robotic systems, where robots that can see, understand, and act in open-ended environments with minimal or absent task-specific training.

### 📊 Validation
This repository contains the codebase and tools for executing the validation of the system on **RoboTHOR** by evaluating VLN and EQA tasks. Specifically, the following tasks have been defined:
- **Vision Language Navigation**
     - Route-oriented tasks (e.g. "Go towards the front door but before the front door make a left, then through the archway, go to the middle of the middle room and stop”)
     - Goal-oriented tasks (e.g. "Find a tv and go toward it")
- **Embodied Question Answering**
     - Color Questions (e.g. "What color is the sofa?")
     - Preposition Questions (e.g. "What is on top of the bed?")
     - Existence Questions (e.g. "Is there a basketball?")
     - Count QUestions (e.g. "How many paintings are in the house?")

## Getting Started
### Installation
Clone the github repository to your local directory:
```bash
git clone https://github.com/tommasoTubaldo/Application_of_VLMs_in_Robotics.git
```

Install all necessary dependencies:
```bash
pip install -qr requirements.txt
pip install -q -U google-genai
```
> **Note**: *Python 3.9+* is required with the google-genai API.

Now, in order to infer with Gemini, you can either use the Google Cloud Services with **Vertex AI** (Reccomended) or the **Gemini API**:

### Vertex AI (Reccomended)

With the Google Cloud Services, you are given $300 of credit to be used with the Vertex AI services, and 90 days of free trial.

First you need to set up a Google Cloud project:
1) Go to https://console.cloud.google.com/projectselector2/home/dashboard
2) Select **"Create a new project"**
3) Choose the project name and select **"Create"**
4) Then, go to https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#confirm_billing_is_enabled_on_a_project and follow the steps to enable the billings for your project
5) Then, enable the Vertex AI API through the following link https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com and by selecting your Google Cloud project

Finally, you need to set up the Google Cloud CLI to interface with the Google Cloud Services:
1) Go to https://cloud.google.com/sdk/docs/install and follow the installation procedure
2) Once installed and intialized, you can login with
   ```bash
   gcloud auth application-default login
   ```

### Gemini API

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
