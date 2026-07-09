import numpy as np
import vtk
import vtkbone
import SimpleITK as sitk

from helper_functions.vtk_util import vtkImageData_to_numpy, numpy_to_vtkImageData
from helper_functions.aim_calibration_header import get_aim_density_equation, get_aim_calibration_constants_from_processing_log


def read_aim_with_position(filename, scaling):
    """
    Read AIM file using vtkbone and return:
    - vtk image
    - numpy array
    - spacing
    - dimensions
    - global AIM position from reader.GetPosition()
    - processing log from reader.GetProcessingLog()
    """
    reader = vtkbone.vtkboneAIMReader()
    reader.DataOnCellsOff()
    reader.SetFileName(filename)
    reader.Update()

    vtk_image = vtk.vtkImageData()
    vtk_image.DeepCopy(reader.GetOutput())

    spacing = np.array(vtk_image.GetSpacing(), dtype=float)
    dims = np.array(vtk_image.GetDimensions(), dtype=int)
    position = np.array(reader.GetPosition(), dtype=int)
    processing_log = reader.GetProcessingLog()

    np_image = vtkImageData_to_numpy(vtk_image)


    print(filename)
    print("  dimensions [voxels]:", dims)
    print("  spacing [mm]:       ", spacing)
    print("  global position:    ", position)
    print("  numpy shape:        ", np_image.shape)
    #print("  processing logs:    ", processing_log)
    print()

    if scaling == 'mu':
        #get calibration information from AIM processing log
        mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = get_aim_calibration_constants_from_processing_log(processing_log)
        np_image_scaled = np_image/mu_scaling
        print("image converted to linear attenuation")
    
    elif scaling == 'HU':
        #get calibration information from AIM processing log
        mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = get_aim_calibration_constants_from_processing_log(processing_log)
        m, b = get_aim_hu_equation(processing_log)
        np_image_scaled = (np_image*m)+b
        print("image converted to HU")
        
    elif scaling == 'BMD':
        #get calibration information from AIM processing log
        mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = get_aim_calibration_constants_from_processing_log(processing_log)
        np_image_scaled = np_image/mu_scaling * density_slope + density_intercept
        print('image converted to bone mineral density')
    
    elif scaling == 'binary':
        np_image_scaled = np_image
        print('image assumed to be binary, values are unchanged')

    elif scaling == 'none': 
        np_image_scaled = np_image
        print("image values are unchanged")   
    
    else:
        raise ValueError(f'{scaling} is not a valid scaling option. Enter with \'HU\', \'mu\', \'BMD\' or \'none\'')
    

    
    
    return {
        "vtk": vtk_image,
        "array": np_image_scaled,
        "units": scaling,
        "spacing": spacing,
        "dims": dims,
        "position": position,
        "processing_log": processing_log,
        "filename": filename,
    }

def paste_mask_into_image_grid(
    mask_array,
    image_shape,
    mask_position,
    image_position,
    binarize=True,
):
    """
    Paste a cropped/local AIM mask into the full greyscale image grid
    using AIM global positions.

    Parameters
    ----------
    mask_array : np.ndarray
        Mask array read from AIM.
        Expected orientation matches vtkImageData_to_numpy output.
    image_shape : tuple
        Shape of target greyscale image array.
    mask_position : array-like
        Global AIM position of the mask from reader.GetPosition().
    image_position : array-like
        Global AIM position of the greyscale image from reader.GetPosition().
    binarize : bool
        If True, convert pasted mask to 0/1.

    Returns
    -------
    aligned_mask : np.ndarray
        Mask in greyscale image grid.
    offset : np.ndarray
        Offset of the mask relative to image grid.
    """

    mask_position = np.array(mask_position, dtype=int)
    image_position = np.array(image_position, dtype=int)

    offset = mask_position - image_position

    print("Image position:", image_position)
    print("Mask position: ", mask_position)
    print("Mask offset in image grid:", offset)

    aligned_mask = np.zeros(image_shape, dtype=np.uint8)

    # Source mask bounds
    src_start = np.array([0, 0, 0], dtype=int)
    src_end = np.array(mask_array.shape, dtype=int)

    # Destination bounds in image grid
    dst_start = offset.copy()
    dst_end = offset + np.array(mask_array.shape, dtype=int)

    # Clip to image grid in case mask partly extends outside image
    for ax in range(3):
        if dst_start[ax] < 0:
            src_start[ax] = -dst_start[ax]
            dst_start[ax] = 0

        if dst_end[ax] > image_shape[ax]:
            src_end[ax] -= dst_end[ax] - image_shape[ax]
            dst_end[ax] = image_shape[ax]

    print("Source crop start:", src_start)
    print("Source crop end:  ", src_end)
    print("Dest paste start: ", dst_start)
    print("Dest paste end:   ", dst_end)

    if np.any(src_end <= src_start) or np.any(dst_end <= dst_start):
        raise ValueError(
            "Mask and image do not overlap after applying global AIM positions."
        )

    mask_crop = mask_array[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]

    if binarize:
        mask_crop = mask_crop > 0

    aligned_mask[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = mask_crop.astype(np.uint8)

    return aligned_mask, offset
	 
def read_nifti_to_numpy_xyz(nifti_file):
    """
    Read a NIfTI image and convert it to x, y, z numpy orientation.

    Use this when the NIfTI already has the same grid/shape as your masks.
    The origin is kept in info for QC, but not used to paste/crop the image.
    """

    nii_img = sitk.ReadImage(nifti_file)

    arr_zyx = sitk.GetArrayFromImage(nii_img)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))

    info = {
        "shape_xyz": arr_xyz.shape,
        "spacing": np.array(nii_img.GetSpacing(), dtype=float),
        "origin": np.array(nii_img.GetOrigin(), dtype=float),
        "direction": np.array(nii_img.GetDirection(), dtype=float).reshape(3, 3),
    }

    print(nifti_file)
    print("  shape xyz:", info["shape_xyz"])
    print("  spacing:  ", info["spacing"])
    print("  origin:   ", info["origin"])
    print("  direction:\n", info["direction"])
    print("  min/max:  ", np.nanmin(arr_xyz), "/", np.nanmax(arr_xyz))
    print("  nonzero:  ", np.count_nonzero(arr_xyz))

    return arr_xyz, info
	 
	 
def calibrate_numpy_image_from_aim_log(
    np_image,
    aim_file,
    scaling="BMD",
):
    """
    Calibrate a NumPy image using the processing log from a specified AIM file.

    Parameters
    ----------
    np_image : np.ndarray
        Image array to calibrate.

    aim_file : str
        Path to AIM file whose processing log contains the calibration constants.

    scaling : str
        Calibration/scaling mode:
        - 'mu'     : native units to linear attenuation
        - 'HU'     : native units to Hounsfield units
        - 'BMD'    : native units to bone mineral density
        - 'binary' : image assumed to be binary, unchanged
        - 'none'   : unchanged

    Returns
    -------
    np_image_scaled : np.ndarray
        Calibrated image.
    """

    reader = vtkbone.vtkboneAIMReader()
    reader.DataOnCellsOff()
    reader.SetFileName(aim_file)
    reader.Update()

    processing_log = reader.GetProcessingLog()

    if scaling == 'mu':
        # get calibration information from AIM processing log
        mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = (
            get_aim_calibration_constants_from_processing_log(processing_log)
        )
        np_image_scaled = np_image / mu_scaling
        print("image converted to linear attenuation")

    elif scaling == 'HU':
        # get calibration information from AIM processing log
        mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = (
            get_aim_calibration_constants_from_processing_log(processing_log)
        )
        m, b = get_aim_hu_equation(processing_log)
        np_image_scaled = (np_image * m) + b
        print("image converted to HU")

    elif scaling == 'BMD':
        # get calibration information from AIM processing log
        mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = (
            get_aim_calibration_constants_from_processing_log(processing_log)
        )
        np_image_scaled = np_image / mu_scaling * density_slope + density_intercept
        print('image converted to bone mineral density')

    elif scaling == 'binary':
        np_image_scaled = np_image
        print('image assumed to be binary, values are unchanged')

    elif scaling == 'none':
        np_image_scaled = np_image
        print("image values are unchanged")

    else:
        raise ValueError(
            f"{scaling} is not a valid scaling option. "
            "Enter with 'HU', 'mu', 'BMD' or 'none'"
        )

    return np_image_scaled

	 
def write_mask_to_aim(
    mask_array,
    output_aim_path,
    reference_aim_path,
    foreground_value=None,
    preserve_labels=True,
):
    """
    Write a NumPy mask or label image to a Scanco .AIM file using vtkbone.

    Parameters
    ----------
    mask_array : np.ndarray
        Mask or label image in x, y, z order.

    output_aim_path : str
        Output AIM path.

    reference_aim_path : str
        Reference AIM file used for spacing, origin, dimensions, and processing log.

    foreground_value : int or None
        If preserve_labels=False, all nonzero voxels are written with this value.
        If None, uses 1.

    preserve_labels : bool
        If True, preserve existing integer label values.
        If False, convert to binary mask.
    """

    # ------------------------------------------------------------
    # Read reference image
    # ------------------------------------------------------------
    reader = vtkbone.vtkboneAIMReader()
    reader.DataOnCellsOff()
    reader.SetFileName(reference_aim_path)
    reader.Update()

    reference_vtk_image = reader.GetOutput()

    spacing = reference_vtk_image.GetSpacing()
    origin = reference_vtk_image.GetOrigin()
    position = reader.GetPosition()
    original_log = reader.GetProcessingLog()

    reference_dims = reference_vtk_image.GetDimensions()

    if mask_array.shape != reference_dims:
        raise ValueError(
            f"mask_array shape {mask_array.shape} does not match "
            f"reference AIM dimensions {reference_dims}"
        )

    # ------------------------------------------------------------
    # Prepare array
    # ------------------------------------------------------------
    if preserve_labels:
        # Preserve labels 0, 1, 2, 3, etc.
        output_array = mask_array.astype(np.int16)
    else:
        if foreground_value is None:
            foreground_value = 1

        output_array = (mask_array > 0).astype(np.int16) * int(foreground_value)

    print("Unique values being written:", np.unique(output_array))

    # ------------------------------------------------------------
    # Convert to vtkImageData
    # ------------------------------------------------------------
    vtk_mask = numpy_to_vtkImageData(
        output_array,
        spacing=spacing,
        origin=origin,
        array_type=vtk.VTK_SHORT,
    )

    processing_log = (
        original_log
        + "\n\n--- Python processing to generate mask ---\n"
        + f"Output file: {output_aim_path}\n"
        + f"Preserve labels: {preserve_labels}\n"
        + f"Unique values: {np.unique(output_array).tolist()}\n"
    )

    # ------------------------------------------------------------
    # Write AIM
    # ------------------------------------------------------------
    writer = vtkbone.vtkboneAIMWriter()
    writer.SetFileName(output_aim_path)
    writer.SetInputData(vtk_mask)

    writer.NewProcessingLogOff()
    writer.SetProcessingLog(processing_log)
    writer.Update()

    print()
    print("Wrote:", output_aim_path)
    print("  dimensions [voxels]:", vtk_mask.GetDimensions())
    print("  resolution [mm]:    ", spacing)
    print("  vtk origin:         ", origin)
    print("  AIM position:       ", position)
    print("  numpy shape:        ", output_array.shape)
    print("  unique values:      ", np.unique(output_array))
    print()