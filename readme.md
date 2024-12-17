
# NER Token Classification with DistilBERT and Gradio UI

This repository contains a Named Entity Recognition (NER) token classification application built using the DistilBERT model. The application provides a simple and intuitive Gradio-based user interface (UI) with two main functionalities:

- Prediction Tab: Input text and get entity classification results in JSON format.

- Configuration & Training Tab: View, update, and finalize model training configurations, and trigger training with real-time logs displayed on the UI.

The application simplifies the training and inference process for NER models by integrating IOB-tagged datasets. 

```python
Please Find Apps screenshot in app_screenshots folder.
```

# Table of Contents

- Features

- Assumptions

- Setup Instructions

- How to Use

- Project Structure

- Technical Details

- Future Improvements

- Acknowledgements


# Features

- Prediction: Input a text string and receive token-level NER predictions in JSON format.

- Configuration Management: Update and finalize training configurations before starting the training process.

- Real-Time Training Logs: Display training progress logs dynamically on the UI.

- IOB-Format Dataset Support: Works seamlessly with BIO-tagged datasets (e.g., restaurant dataset).

# Assumptions - (IMPORTANT)

1. The application takes IOB (.bio) format files as input for training. You may use any such dataset; this project uses the "restaurant" dataset for demonstration.

2. The Configs are not protected, any absured configuration given may lead to failure. Please take precautions before entering any configuration. 

3. Real-time training logs are enabled for better visibility using a global variable. For cleaner modularization (without real-time logging), the trainer code can be moved back to a separate script in the source_code folder.

4. This is the first version of the UI, so minor glitches may exist. Feedback is welcome for improvements.

# Model training output in terminal

| **Metric**    | **Value**    |
| :-------- | :------- |
| **overall-f1** | `79.2%` |
| **overall-accuracy** | `91.4%` |
| **overall-precision** | `77.4%` |
| **overall-recall** | `81.1%` |



# Setup Instructions

## Follow these steps to set up the application:

1. ### Clone the Repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. ### Set Up Virtual Environment: Ensure you have Python 3.11 installed.

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. ### Install Dependencies:
Run the requirements.txt file to install all necessary packages.

```bash
pip install -r requirements.txt
```

4. ### Run the Application:
Launch the Gradio UI by executing the app.py script.

```bash
python app.py
```

5. ### Access the UI:
Open the application in your browser at:
```bash
http://localhost:7860
```

# How to Use

## 1. Prediction Tab

    - Input any text string in the provided field.

    - Click the "Submit" button to get NER token classification predictions.

    - Results are displayed in a JSON editor format for clarity.

## 2. Configuration & Training Tab

    - Review and update model training configurations as needed.

    - Check the "Finalize Configuration" checkbox to enable training.

    - Once finalized, click "Start Training" to trigger the training process.

    - Monitor real-time training logs directly on the UI.

# Technical Details

- Model: DistilBERT fine-tuned for token-level NER classification.

- Dataset: IOB (.bio) format files are used as input for training.

- Frameworks/Libraries:

    - Hugging Face Transformers

    - Gradio for the UI.

    - Python 3.11

# Future Improvements

- Modularize the training script for cleaner code structure.

- Add better UI responsiveness and error handling.

- Support multiple datasets with upload functionality.

- Implement a database to persist configurations and model checkpoints.

- Add an option for model evaluation with validation metrics.

# Acknowledgements

- Thanks to the Hugging Face community for the excellent Transformers library.

- Special credit to Gradio for making ML application UIs easy to build.

