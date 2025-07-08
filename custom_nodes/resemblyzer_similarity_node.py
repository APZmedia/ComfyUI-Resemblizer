from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

class APZmediaResemblyzerSimilarityNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1_path": ("STRING", {"default": ""}),
                "audio2_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE_PATH",)
    RETURN_NAMES = ("similarity_score", "similarity_plot",)
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

            # Framewise similarity plot
            partials1 = encoder.embed_utterance(wav1, return_partials=True)
            partials2 = encoder.embed_utterance(wav2, return_partials=True)
            min_len = min(partials1.shape[0], partials2.shape[0])
            partials1 = partials1[:min_len]
            partials2 = partials2[:min_len]
            similarities = np.sum(partials1 * partials2, axis=1) / (
                np.linalg.norm(partials1, axis=1) * np.linalg.norm(partials2, axis=1)
            )

            # Plot and save to a temp file
            plt.figure(figsize=(10, 4))
            plt.plot(similarities, label="Cosine Similarity")
            plt.title("Framewise Similarity between Two Audios")
            plt.xlabel("Frame")
            plt.ylabel("Cosine Similarity")
            plt.ylim(-1, 1)
            plt.legend()
            plt.tight_layout()
            tmpdir = tempfile.gettempdir()
            plot_path = os.path.join(tmpdir, "apzmedia_similarity_plot.png")
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"Error processing audio files: {e}")
            similarity = -1.0  # Indicates error
            plot_path = ""
        return (similarity, plot_path)

NODE_CLASS_MAPPINGS = {
    "APZmediaResemblyzerSimilarityNode": APZmediaResemblyzerSimilarityNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APZmediaResemblyzerSimilarityNode": "APZmedia Resemblyzer Audio Similarity",
} 