# Resemblyzer Audio Similarity Node for ComfyUI

This custom node allows you to compare the similarity of two audio files using the [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) deep learning model. It outputs a similarity score based on the cosine similarity of the voice embeddings.

## Features
- Accepts two audio file paths as input
- Uses Resemblyzer to extract voice embeddings
- Outputs a similarity score (float)

## Installation
1. Place this folder (`custom_nodes`) inside your ComfyUI root directory if not already present.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Restart ComfyUI after adding this node.
- In the ComfyUI interface, search for **Resemblyzer Audio Similarity** under the Audio category.
- Provide two audio file paths as input.
- The node will output a similarity score between -1.0 and 1.0 (higher means more similar).

## Example
```python
# Example usage in Python (not in ComfyUI UI):
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

wav1 = preprocess_wav(Path("audio1.wav"))
wav2 = preprocess_wav(Path("audio2.wav"))
encoder = VoiceEncoder()
embed1 = encoder.embed_utterance(wav1)
embed2 = encoder.embed_utterance(wav2)
similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
print(similarity)
```

## Requirements
See `requirements.txt` for dependencies.

## License
Apache-2.0 (same as Resemblyzer) 