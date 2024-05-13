from typing import List, Tuple, Type
import pydicom
import numpy as np
from pydicom.dicomdir import DicomDir

def get_window(dcm: DicomDir) -> Tuple[int, int]:
    """Returns default windowing of a dicom image.

    Returns default windowing values of the dcm store, if present.
    If not present, compute the window center and width as mean and std 
    of the pixel_array distribution. 

    Args:
        dcm (DicomDir): input dicom

    Returns:
        tuple[int, int]: tuple containing (window_center, window_width)
    """
    if 'WindowCenter' not in dcm or 'WindowWidth' not in dcm:
        window_center, window_width = int(np.mean(dcm.pixel_array)), int(np.std(dcm.pixel_array)*2)
    
    else:
        window_center = dcm.WindowCenter
        window_width = dcm.WindowWidth

    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = window_center[0]

    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = window_width[0]

    if window_width == 0:
        window_width = 1

    return window_center, window_width

def get_full_window(dcm: DicomDir) -> Tuple[int, int]:
    """Compute full-dynamic windowing for a dcm.

    Args:
        dcm (DicomDir): input dicom

    Returns:
        tuple[int, int]: tuple containing (window_center, window_width)
    """
    value_min, value_max = dcm.pixel_array.min(), dcm.pixel_array.max()
    window_width = (value_max - value_min)
    window_center = value_min + (window_width // 2)
    return window_center, window_width

def make_lut(dcm: DicomDir, window_center: int=None, window_width: int=None, bpp: int=8) -> List[int]:
    """Build a linear VOI LUT, which also includes a modality LUT, with the specified pixel depth.

    Examples:
        ```python
        dcm = pydicom.read_file(filename)
        lut = make_lut(dcm, bpp=16)
        ```
    Args:
        dcm (DicomDir): input dicom
        window_center (int, optional): If None, use dcm default window center. Defaults to None.
        window_width (int, optional): If None, use dcm default window width. Defaults to None.
        bpp (int, optional):Desired pixel depth in the resulting lut. Defaults to 8.

    Returns:
        List[int]: linear lut
    """
    
    # Parameters for the modality LUT
    slope = dcm.get('RescaleSlope', 1)
    intercept = dcm.get('RescaleIntercept', 0)

    if window_center is None:
        window_center, _ = get_window(dcm)
    
    if window_width is None:
        _, window_width = get_window(dcm)

    storedPixels = dcm.pixel_array.copy()
    minPixel = int(np.amin(storedPixels))
    maxPixel = int(np.amax(storedPixels))
    
    # Invert the specified window_center for MONOCHROME1, so that increasing 
    # the level value makes the images brighter regardless of photometric intrepretation
    invert = False
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        invert = True
        # window_center = (maxPixel - minPixel) - window_center 

    # Calculate LUT
    gray_levels = float((2**bpp)-1)
    lut = [0] * (maxPixel + 1)

    for storedValue in range(minPixel, maxPixel):
        modalityLutValue = storedValue * slope + intercept

        # The window is built as [center - width/2; center + width/2], so we add 0.5
        voiLutValue = (((modalityLutValue - window_center) / window_width + 0.5) * gray_levels)
        
        # Clamp the final value between 0 and (2^bpp)-1
        clampedValue = min(max(voiLutValue, 0), gray_levels)

        if invert:
            lut[storedValue] = round(gray_levels-clampedValue)
        else:
            lut[storedValue] = round(clampedValue)

    lut[0] = 0
    if len(lut) > 1:    
        lut[-1] = lut[-2]
    return lut

def apply_lut(pixels_in: np.array, lut: List[int], dtype: Type[np.unsignedinteger]=np.uint8) -> np.array:
    """Apply a LUT to a pixel array.

    Examples:
        ```python
        dtypes = {
            8: np.uint8,
            16: np.uint16,
            32: np.uint32,
            64: np.uint64
        }
        bpp = 16

        dcm = pydicom.read_file(filename)
        lut = make_lut(dcm, bpp=bpp)
        widowed = apply_lut(dcm.pixels_array, lut, dtype=dtypes[bpp])
        ```

    Args:
        pixels_in (np.array): input pixels
        lut (List[int]): computed LUT
        dtype (Type[np.unsignedinteger], optional): Pixel depth. Defaults to np.uint8.

    Returns:
        np.array: transformed pixels.
    """
    shape = pixels_in.shape
    pixels_in = pixels_in.copy().flatten()
    pixels_out = np.zeros_like(pixels_in, dtype=dtype)
    
    for idx, pixel in enumerate(pixels_in.flatten()):
        pixels_out[idx] = lut[pixel]
        
    return pixels_out.reshape(shape)