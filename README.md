<div align="center">

# PPE Detection with YOLO

**Real-time computer vision prototype for detecting personal protective equipment in webcam frames.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-111827)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

</div>

## Overview

This project runs a trained YOLO object-detection model on a webcam stream and overlays detected PPE classes with bounding boxes and confidence scores. It is an **experimental computer-vision project**, not a certified workplace-safety monitoring system.

### Detection classes

| Class in model | English |
| --- | --- |
| \`capacete\` | Helmet |
| \`colete\` | Safety vest |
| \`luvas\` | Gloves |
| \`mascara\` | Mask |
| \`oculos\` | Safety glasses |

The class ordering in \`CLASS_NAMES\` must match the labels used when training the supplied \`.pt\` model.

## How it works

1. Open webcam device \`0\` with OpenCV.
2. Load \`datsetpropriov2.pt\` through Ultralytics YOLO.
3. Run inference on each camera frame.
4. Ignore detections below the configured **0.60 confidence threshold**.
5. Draw bounding boxes, class labels, confidence scores and a summary of detected classes.
6. Press **Q** to end the stream.

> **Important limitation:** Detecting an item in the frame does not prove that a specific person is wearing every required item correctly. This code does not establish individual PPE compliance or validate production reliability.

## Quick start

**Requirements:** Python 3, an accessible webcam, and dependencies supported by your Python/OS environment.

\`\`\`bash
git clone https://github.com/LucasDiasRamos/IA_EPI.git
cd IA_EPI
python -m venv .venv
\`\`\`

Activate your virtual environment:

- Linux/macOS: \`source .venv/bin/activate\`
- Windows PowerShell: \`.venv\Scripts\Activate.ps1\`

Install the libraries:

\`\`\`bash
pip install ultralytics opencv-python
\`\`\`

Run:

\`\`\`bash
python main.py
\`\`\`

The custom model file is included in the repository as \`datsetpropriov2.pt\`. Run the script from the repository root so relative paths resolve correctly.

## Configuration

At the top of \`main.py\`, edit the relevant settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| \`WEBCAM_ID\` | \`0\` | Camera index; can be adjusted for virtual/secondary cameras |
| \`MODEL_PATH\` | \`datsetpropriov2.pt\` | YOLO weights file |
| \`MIN_CONFIDENCE\` | \`0.60\` | Detection confidence cutoff |
| \`CLASS_NAMES\` | 5 PPE classes | Labels matching the trained model |
| \`DISPLAY_WIDTH\`, \`DISPLAY_HEIGHT\` | 1920, 1080 | Display window dimensions |

The camera capture itself is requested at 640 × 480; the display window is resized separately.

## Repository structure

\`\`\`text
IA_EPI/
├── main.py                 # Real-time PPE detection
├── validacao_epis.py       # Alternate validation script
├── teste_gpu.py            # GPU-related test script
└── datsetpropriov2.pt      # Project model weights
\`\`\`

## Demo and future improvements

A verified screenshot or demo video has not yet been added. Useful next steps would be to add a short recording, document model evaluation results (precision/recall, dataset provenance and limitations), improve configuration through CLI arguments, and introduce automated tests where practical.

## Author

[Lucas Dias](https://github.com/LucasDiasRamos)
