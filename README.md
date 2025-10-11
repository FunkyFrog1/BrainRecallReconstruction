# Brain Recall Reconstruction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

A deep learning framework for decoding EEG signals from human recall of visual images and reconstructing recognizable pictures. This project integrates neuroscience and computer vision to explore direct image generation from brain activity.

## Project Overview

- **Objective**: Extract features from EEG signals and use deep learning models to reconstruct visual images recalled by subjects (e.g., objects, scenes).
- **Key Innovations**:
  - Pioneering reconstruction of images directly from short-term human memory using EEG signals.
  - End-to-end deep learning architecture that maps EEG data to image embeddings for efficient and interpretable reconstruction.
- **Applications**: Brain-Computer Interfaces (BCI), neuroscience research, medical diagnostics (e.g., memory disorder analysis), AI-driven creative tools.
- **Status**: Actively developed as a thesis project, with plans for future publication.

## Features

- **Data Preprocessing**: Support for EEG signal filtering, denoising, time-frequency analysis (e.g., wavelet transform), and alignment with image data.
- **Model Architecture**: Modular deep learning models for flexible experimentation.
- **Training Pipeline**: Configurable training scripts with support for various loss functions and metrics.
- **Evaluation**: Tools for quantitative and qualitative assessment of reconstructed images.
- **Visualization**: Utilities to visualize EEG signals, model outputs, and intermediate features.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- Other dependencies as listed in `requirements.txt`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brain-recall-reconstruction.git
   cd brain-recall-reconstruction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment.

## Usage (Updating)

1. **Prepare Data**: Download EEG and image data in the `data/` directory. 
2. **Preprocess Vsion Backbone**: Download Vision Backbone checkpoint in the `vision_backbone/` directory.
3. **Install Requirement**
4. **Train Model**:
   run `scr/train.py`
5. **Reconstruct Images**:
   updating
6. **Evaluate Results**: Use provided scripts to compute metrics and generate visualizations.
   updating
For detailed instructions, see the [documentation](docs/).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

By using this software, you agree to comply with the license terms. If you use this code in your research, please cite the associated paper (to be added upon publication).


## Citation

If you use this code in your research, please cite:

```
@article{your2025brain,
  title={Brain Recall Reconstruction: Decoding EEG Signals for Image Generation},
  author={Your Name and Others},
  journal={To be published},
  year={2025}
}
```

(Not yet public)

## Contact

For questions or collaborations, please contact [Frog](mailto:2023024424@m.scnu.edu.cn) or open an issue on GitHub.

## Acknowledgments

- Thanks to advisors and collaborators.
- Inspired by prior work in EEG decoding and image reconstruction.
