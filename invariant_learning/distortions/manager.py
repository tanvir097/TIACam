from augmentors.filtering.inference import apply_filtering
from augmentors.jpeg.inference import apply_jpeg
from augmentors.moire.inference import apply_moire
from augmentors.perspective.inference import apply_perspective
from augmentors.photometric.inference import apply_photometric
from augmentors.photometric.inference import apply_additive


def build_distortions():
    return [
        apply_additive,
        apply_filtering,
        apply_jpeg,
        apply_moire,
        apply_perspective,
        apply_photometric
    ]