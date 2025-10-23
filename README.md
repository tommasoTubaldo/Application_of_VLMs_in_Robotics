# Application of VLMs in Robotics

This project demonstrates how Vision Language Models (VLMs) can enhance robotic systems through vision understanding and natural language interaction. It implements a function-calling interface that connects VLMs to robotic perception/control modules for two core tasks: Room-to-Room Navigation (R2R) and Embodied Question Answering (EQA). The system is evaluated in AI2-THOR simulation, showing how foundation models can improve robotic perception and planning.

## Demo
https://github.com/user-attachments/assets/77a83a23-825e-49d2-8555-c938ee22f736

## Getting Started
### Prerequisites
- **Python 3.9+** (python 3.11.5 was used for the development).

### Installation
Clone the repository:
```bash
git clone https://github.com/tommasoTubaldo/Application_of_VLMs_in_Robotics.git
cd Application_of_VLMs_in_Robotics
```

Install dependencies:
```bash
pip install -r requirements.txt
pip install -U google-genai
```
> **Note**: Use a virtual environment (```python -m venv venv``` and then ```source venv/bin/activate```) to avoid conflicts.

## Configuration

Choose either **Vertex AI** (recommended) or **Gemini API**:

### Option 1: Vertex AI (Reccomended)
With the Google Cloud Services, you are given $300 of credit to be used with the Vertex AI services, and 90 days of free trial.

1. Set Up Google Cloud Project:
     - Go to [Google Cloud Console](https://console.cloud.google.com/projectselector2/home/dashboard).
     -  Click **"Create a new project"**, name it and confirm.
     -   Enable Billing:
          -   Follow [this guide](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#confirm_billing_is_enabled_on_a_project) to link a billing account.
     -  Enable Vertex AI API:
          -  Visit the [API enablement page](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com) and select your project.

2. Install the Google Cloud CLI:
     - Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).
     -  Authenticate and log in:
        ```bash
        gcloud auth application-default login
        ```
   
3. Configure environment variables:
   - Run these commands on your project directory:
        ```bash
        export PROJECT_ID="<your_project_id>"
        export LOCATION="<your_location>"
        export API_MODE="vertex"
        ```
   
   - You can find the project id by entering the [Google Cloud Console](https://console.cloud.google.com/) and following [these instructions](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction/prerequisites).
   - You can choose the location by refering to [Vertex AI regions](https://cloud.google.com/vertex-ai/docs/general/locations).

### Option 2: Gemini API (Simpler, but rate-limited)

1. Get an API key:
   - Go to [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
   - Click **"Create API key"** and copy the key.
     
2. Configure environment:
   - In your project directory:
        ```bash
        cd ~/your_project
        export GEMINI_API_KEY="<your_api_key>"
        export API_MODE="gemini"
        ```

## Run the project
To run the project, simply run:
```bash
python3 main.py
```
