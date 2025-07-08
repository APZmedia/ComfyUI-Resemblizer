from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

class ResemblyzerSimilarityNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1_path": ("STRING", {"default": ""}),
                "audio2_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("similarity_score",)
    FUNCTION = "compare_audios"
    CATEGORY = "Audio"

    def compare_audios(self, audio1_path, audio2_path):
        try:
            wav1 = preprocess_wav(Path(audio1_path))
            wav2 = preprocess_wav(Path(audio2_path))
            encoder = VoiceEncoder()
            embed1 = encoder.embed_utterance(wav1)
            embed2 = encoder.embed_utterance(wav2)
            similarity = float(np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2)))
        except Exception as e:
            print(f"Error processing audio files: {e}")
            similarity = -1.0  # Indicates error
        return (similarity,)

NODE_CLASS_MAPPINGS = {
    "ResemblyzerSimilarityNode": ResemblyzerSimilarityNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResemblyzerSimilarityNode": "Resemblyzer Audio Similarity",
} 