from data.preprocessing import ImagePreprocessing
from data.processing import FrangiFilter2D
from data.postprocessing import ImagePostprocessing
import numpy as np

class FilterSegmentation:
    @staticmethod
    def run(image: np.ndarray) -> np.ndarray:
        # green = ImagePreprocessing.extract_green_channel(image) # alternative
        gray = ImagePreprocessing.grayscale(image)
        clahe = ImagePreprocessing.apply_clahe(gray)
        median_filtered = ImagePreprocessing.median_filter(clahe)
        normalized = ImagePreprocessing.normalize(median_filtered)
        resized_normalized = ImagePreprocessing.resize_and_normalize(normalized)
        frangi_result = FrangiFilter2D.apply(resized_normalized)
        binary = ImagePostprocessing.adaptive_binarization(frangi_result)
        opened = ImagePostprocessing.morphological_opening(binary)
        closed = ImagePostprocessing.morphological_closing(opened)

        negative = 255 - closed

        return negative
