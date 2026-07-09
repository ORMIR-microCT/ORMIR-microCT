import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label
import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import HBox, VBox
from IPython.display import display, clear_output
from matplotlib.colors import ListedColormap

def largest_component_3d(mask, connectivity=1):
    """
    Keep the largest connected component in a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary 3D mask.
    connectivity : int
        1 = face connectivity, 2 or 3 = more permissive connectivity.

    Returns
    -------
    largest : np.ndarray
        Binary mask containing only the largest component.
    """
    mask = mask.astype(bool)

    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
    labels, n_labels = ndi.label(mask, structure=structure)

    if n_labels == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    counts = np.bincount(labels.ravel())
    counts[0] = 0

    largest_label = counts.argmax()
    largest = labels == largest_label

    return largest.astype(np.uint8)
	 
def get_padded_bbox(mask, pad=0):
    """
    Get a padded bounding box around nonzero voxels in a 3D mask.
    """
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        raise ValueError("Mask is empty; cannot compute bounding box.")

    start = coords.min(axis=0) - pad
    end = coords.max(axis=0) + 1 + pad

    start = np.maximum(start, 0)
    end = np.minimum(end, mask.shape)

    bbox = tuple(slice(start[i], end[i]) for i in range(3))

    return bbox, {
        "start": start,
        "end": end,
        "shape": end - start,
    }


def fast_binary_dilate(mask, radius_vox):
    """
    Fast spherical binary dilation using Euclidean distance transform.
    """
    mask = mask.astype(bool)

    if radius_vox <= 0:
        return mask.copy()

    return ndi.distance_transform_edt(~mask) <= radius_vox


def fast_binary_erode(mask, radius_vox):
    """
    Fast spherical binary erosion using Euclidean distance transform.
    """
    mask = mask.astype(bool)

    if radius_vox <= 0:
        return mask.copy()

    return ndi.distance_transform_edt(mask) > radius_vox


def connectivity_aware_closing_fast(input_mask,search_mask=None,radius_vox=10,connectivity=1,return_intermediates=False):
    """
    Connectivity-aware closing using distance-transform morphology.

    Workflow:
    1. Dilate input_mask.
    2. Restrict to search_mask.
    3. Invert within search_mask.
    4. Keep largest inverse component.
    5. Invert that component.
    6. Erode back.
    """

    input_mask = input_mask.astype(bool)

    if search_mask is None:
        search_mask = np.ones_like(input_mask, dtype=bool)
    else:
        search_mask = search_mask.astype(bool)

    if input_mask.shape != search_mask.shape:
        raise ValueError(
            f"input_mask and search_mask must have the same shape, "
            f"got {input_mask.shape} and {search_mask.shape}"
        )

    dilated_mask = fast_binary_dilate(
        input_mask,
        radius_vox=radius_vox,
    )

    dilated_mask = dilated_mask & search_mask

    inverse_region = search_mask & (~dilated_mask)

    largest_inverse_component = largest_component_3d(
        inverse_region,
        connectivity=connectivity,
		  ).astype(bool)

    filled_or_compartment_dilated = search_mask & (~largest_inverse_component)

    output_mask = fast_binary_erode(
        filled_or_compartment_dilated,
        radius_vox=radius_vox,
		  )

    output_mask = output_mask & search_mask

    outputs = {
        "output_mask": output_mask.astype(np.uint8),
	  }

    if return_intermediates:
        outputs.update(
            {
                "dilated_mask": dilated_mask.astype(np.uint8),
                "inverse_region": inverse_region.astype(np.uint8),
                "largest_inverse_component": largest_inverse_component.astype(np.uint8),
                "filled_or_compartment_dilated": filled_or_compartment_dilated.astype(np.uint8),
            }
				
				)

    return outputs


def connectivity_aware_closing_fast_cropped(input_mask,search_mask,radius_vox=10,connectivity=1,crop_pad_vox=None,return_intermediates=False,verbose=True):
    """
    Cropped connectivity-aware closing using distance transforms.

    This is the efficient version for generating trabecular compartment masks
    from local marrow masks.

    Parameters
    ----------
    input_mask : np.ndarray
        Binary input mask, for example main_marrow or epiphysis_marrow.

    search_mask : np.ndarray
        Binary domain mask, usually whole_bone.

    radius_vox : int
        Dilation/erosion radius in voxels.

    connectivity : int
        Connected-component connectivity.

    crop_pad_vox : int or None
        Padding around input_mask bounding box.
        If None, uses 4 * radius_vox.

    return_intermediates : bool
        If True, returns intermediate masks pasted into full image space.
        Keep False for speed.

    verbose : bool
        If True, print crop information.

    Returns
    -------
    outputs : dict
        - output_mask
        - bbox
        - bbox_bounds
        - optional intermediates
    """

    input_mask = input_mask.astype(bool)
    search_mask = search_mask.astype(bool)

    if input_mask.shape != search_mask.shape:
        raise ValueError(
            f"input_mask and search_mask must have the same shape, "
            f"got {input_mask.shape} and {search_mask.shape}"
        )

    if crop_pad_vox is None:
        crop_pad_vox = int(4 * radius_vox)

    bbox, bounds = get_padded_bbox(
        input_mask,
        pad=crop_pad_vox,
    )

    input_crop = input_mask[bbox]
    search_crop = search_mask[bbox]

    if verbose:
        print("Full shape:", input_mask.shape)
        print("Crop start:", bounds["start"])
        print("Crop end:  ", bounds["end"])
        print("Crop shape:", bounds["shape"])

    crop_outputs = connectivity_aware_closing_fast(
        input_mask=input_crop,
        search_mask=search_crop,
        radius_vox=radius_vox,
        connectivity=connectivity,
        return_intermediates=return_intermediates,
    )

    output_full = np.zeros_like(input_mask, dtype=np.uint8)
    output_full[bbox] = crop_outputs["output_mask"].astype(np.uint8)

    outputs = {
        "output_mask": output_full,
        "bbox": bbox,
        "bbox_bounds": bounds,
    }

    if return_intermediates:
        for key, value in crop_outputs.items():
            if key == "output_mask":
                continue

            full = np.zeros_like(input_mask, dtype=np.uint8)
            full[bbox] = value.astype(np.uint8)
            outputs[key] = full

    return outputs

_BALL_KERNEL_CACHE = {}

def ball_kernel(radius_vox):
    """
    Create and cache a 3D spherical structuring element.

    radius_vox <= 0 returns None.
    """
    r = int(radius_vox)

    if r <= 0:
        return None

    if r in _BALL_KERNEL_CACHE:
        return _BALL_KERNEL_CACHE[r]

    x, y, z = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    kernel = (x**2 + y**2 + z**2) <= r**2

    _BALL_KERNEL_CACHE[r] = kernel

    return kernel

def binary_morphology_ball(
    mask,
    operation,
    radius_vox,
    use_distance_transform=False):
    """
    Apply binary morphology with a spherical radius.

    operation:
    - "dilate"
    - "erode"
    - "close"
    """

    mask = mask.astype(bool)

    if radius_vox <= 0:
        return mask.copy()

    operation = operation.lower()

    if operation not in {"dilate", "erode", "close"}:
        raise ValueError("operation must be 'dilate', 'erode', or 'close'")

    if use_distance_transform:
        if operation == "dilate":
            return ndi.distance_transform_edt(~mask) <= radius_vox

        if operation == "erode":
            return ndi.distance_transform_edt(mask) > radius_vox

        if operation == "close":
            dilated = ndi.distance_transform_edt(~mask) <= radius_vox
            return ndi.distance_transform_edt(dilated) > radius_vox

    kernel = ball_kernel(radius_vox)

    if operation == "dilate":
        return ndi.binary_dilation(mask, structure=kernel)

    if operation == "erode":
        return ndi.binary_erosion(mask, structure=kernel)

    if operation == "close":
        return ndi.binary_closing(mask, structure=kernel)
        

def extract_marrow_from_bone(
    seg_bone,
    whole_bone,
    connectivity=1,
    bone_closing_radius_vox=1,
    whole_bone_inner_margin_vox=1,
    use_distance_transform=False,
    label_main=1,
    label_epiphysis=2,
    return_info=True,
):
    """
    Extract main marrow and epiphysis marrow from a segmented bone image.

    Returns a single labeled image instead of separate binary masks.

    Output labels
    -------------
    0 = background
    1 = main marrow compartment
    2 = epiphysis marrow compartment

    Workflow
    --------
    1. Close the low-threshold segmented bone to seal small gaps.
    2. Constrain the closed bone segmentation to the whole-bone mask.
    3. Erode the whole-bone mask slightly to remove peripheral rim artifacts.
    4. Invert closed bone segmentation inside the inner whole-bone domain.
    5. Label connected non-bone components.
    6. Assign the largest component to label_main.
    7. Assign the second-largest component to label_epiphysis.

    Parameters
    ----------
    seg_bone : np.ndarray
        Binary segmented bone image.

    whole_bone : np.ndarray
        Binary whole-bone mask, aligned to the greyscale image.

    connectivity : int
        Connectivity used by scipy.ndimage.label.

    bone_closing_radius_vox : int
        Radius for closing of seg_bone. Use 0 to skip.

    whole_bone_inner_margin_vox : int
        Erode whole_bone by this many voxels before extracting components.
        Use 0 to skip.

    use_distance_transform : bool
        If True, use distance-transform morphology.

    label_main : int
        Label value assigned to the largest marrow component.

    label_epiphysis : int
        Label value assigned to the second-largest marrow component.

    return_info : bool
        If True, return diagnostic information in addition to the labeled image.

    Returns
    -------
    marrow_labels : np.ndarray
        Labeled marrow image, uint8.

    info : dict, optional
        Diagnostic information.
    """

    seg_bone = seg_bone.astype(bool)
    whole_bone = whole_bone.astype(bool)

    if seg_bone.shape != whole_bone.shape:
        raise ValueError(
            f"seg_bone and whole_bone must have the same shape, "
            f"got {seg_bone.shape} and {whole_bone.shape}"
        )

    # 1) Close segmented bone
    seg_bone_closed = binary_morphology_ball(
        seg_bone,
        operation="close",
        radius_vox=bone_closing_radius_vox,
        use_distance_transform=use_distance_transform,
    )

    seg_bone_closed &= whole_bone

    # 2) Erode whole-bone search domain to remove peripheral rim artifacts
    whole_bone_inner = binary_morphology_ball(
        whole_bone,
        operation="erode",
        radius_vox=whole_bone_inner_margin_vox,
        use_distance_transform=use_distance_transform,
    )

    # 3) Invert closed bone segmentation inside inner whole-bone domain
    non_bone_inside = whole_bone_inner & (~seg_bone_closed)

    # 4) Connected components
    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
    labels, n_labels = ndi.label(non_bone_inside, structure=structure)

    marrow_labels = np.zeros(seg_bone.shape, dtype=np.uint8)

    if n_labels == 0:
        if return_info:
            info = {
                "component_volumes_vox": [],
                "main_component_label": None,
                "epiphysis_component_label": None,
                "main_label_value": label_main,
                "epiphysis_label_value": label_epiphysis,
                "n_components": 0,
            }
            return marrow_labels, info

        return marrow_labels

    # Component sizes, excluding background
    counts = np.bincount(labels.ravel())
    component_labels = np.arange(1, len(counts))
    component_sizes_vox = counts[1:]

    sort_idx = np.argsort(component_sizes_vox)[::-1]
    sorted_component_labels = component_labels[sort_idx]
    sorted_component_sizes_vox = component_sizes_vox[sort_idx]

    # Largest component = main marrow
    main_component_label = int(sorted_component_labels[0])
    marrow_labels[labels == main_component_label] = label_main

    # Second-largest component = epiphysis marrow
    if len(sorted_component_labels) >= 2:
        epiphysis_component_label = int(sorted_component_labels[1])
        marrow_labels[labels == epiphysis_component_label] = label_epiphysis
    else:
        epiphysis_component_label = None

    if return_info:
        info = {
            "component_volumes_vox": sorted_component_sizes_vox.tolist(),
            "main_component_label": main_component_label,
            "epiphysis_component_label": epiphysis_component_label,
            "main_label_value": label_main,
            "epiphysis_label_value": label_epiphysis,
            "n_components": int(n_labels),
            "main_voxels": int(np.count_nonzero(marrow_labels == label_main)),
            "epiphysis_voxels": int(np.count_nonzero(marrow_labels == label_epiphysis)),
            "unique_values": np.unique(marrow_labels).tolist(),
        }

        print("Unique marrow labels:", info["unique_values"])
        print("Main marrow voxels:", info["main_voxels"])
        print("Epiphysis marrow voxels:", info["epiphysis_voxels"])
        print("Component volumes [vox]:", info["component_volumes_vox"][:10])

        return marrow_labels, info

    return marrow_labels	  


def segment_mineralized_bone_compartments(
    image,
    whole_bone_mask,
    compartment_seg,
    bone_threshold=300,
    gaussian_sigma_vox=0.8,
    label_cortical=1,
    label_trabecular=2,
    label_epiphysis=3,
    return_info=True,
):
    """
    Segment mineralized bone and assign compartment labels.

    Input compartment labels
    ------------------------
    compartment_seg:
        0 = outside compartment
        1 = main trabecular compartment
        2 = epiphysis compartment

    Output bone labels
    ------------------
    bone_labels:
        0 = background / non-mineralized
        1 = cortical bone
        2 = trabecular bone
        3 = epiphyseal bone

    Returns
    -------
    bone_labels : np.ndarray
        Labeled mineralized bone image.

    info : dict, optional
        Small metadata dictionary with voxel counts and label values.
    """

    image = np.asarray(image)
    whole_bone_mask = np.asarray(whole_bone_mask).astype(bool)
    compartment_seg = np.asarray(compartment_seg)

    if not (image.shape == whole_bone_mask.shape == compartment_seg.shape):
        raise ValueError(
            "image, whole_bone_mask, and compartment_seg must have the same shape."
        )

    if gaussian_sigma_vox and gaussian_sigma_vox > 0:
        image_smooth = ndi.gaussian_filter(
            image.astype(np.float32, copy=False),
            sigma=gaussian_sigma_vox,
        )
    else:
        image_smooth = image.astype(np.float32, copy=False)

    # Mineralized bone segmentation.
    bone_seg = (image_smooth >= bone_threshold) & whole_bone_mask

    # Single output label image.
    bone_labels = np.zeros(image.shape, dtype=np.uint8)

    # Cortical first: all mineralized bone.
    bone_labels[bone_seg] = label_cortical

    # Then overwrite with compartment-specific labels.
    bone_labels[bone_seg & (compartment_seg == 1)] = label_trabecular
    bone_labels[bone_seg & (compartment_seg == 2)] = label_epiphysis

    # Release large temporary arrays explicitly.
    del bone_seg
    if gaussian_sigma_vox and gaussian_sigma_vox > 0:
        del image_smooth

    if not return_info:
        return bone_labels

    info = {
        "labels": {
            "background": 0,
            "cortical": label_cortical,
            "trabecular": label_trabecular,
            "epiphysis": label_epiphysis,
        },
        "unique_values": np.unique(bone_labels).tolist(),
        "cortical_voxels": int(np.count_nonzero(bone_labels == label_cortical)),
        "trabecular_voxels": int(np.count_nonzero(bone_labels == label_trabecular)),
        "epiphyseal_voxels": int(np.count_nonzero(bone_labels == label_epiphysis)),
        "total_bone_voxels": int(np.count_nonzero(bone_labels > 0)),
    }

    print("Unique bone labels:", info["unique_values"])
    print("Cortical bone voxels:   ", info["cortical_voxels"])
    print("Trabecular bone voxels: ", info["trabecular_voxels"])
    print("Epiphyseal bone voxels: ", info["epiphyseal_voxels"])
    print("Total bone voxels:      ", info["total_bone_voxels"])

    return bone_labels, info


def combine_masks_with_labels(
    masks,
    labels,
    dtype=np.uint8,
):
    """
    Combine binary masks into one labeled array.

    Later masks overwrite earlier masks if they overlap.

    Parameters
    ----------
    masks : list of np.ndarray
        Binary masks to combine.

    labels : list of int
        Label value assigned to each mask.

    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    labeled : np.ndarray
        Labeled mask array.
    """

    if len(masks) != len(labels):
        raise ValueError("masks and labels must have the same length.")

    reference_shape = masks[0].shape

    labeled = np.zeros(reference_shape, dtype=dtype)

    for mask, label in zip(masks, labels):
        if mask.shape != reference_shape:
            raise ValueError("All masks must have the same shape.")

        labeled[np.asarray(mask).astype(bool)] = label

    return labeled




def extract_compartment_region_by_bone_length(
    compartment_mask,
    whole_bone_mask,
    label_to_use,
    percent_length=10,
    axis=2,
    start_from="max",
):
    """
    Extract a fixed-length subregion from one label of a compartment mask.

    The region length is calculated as a percentage of the full bone length,
    then applied to the selected label in the compartment mask.

    Parameters
    ----------
    compartment_mask : np.ndarray
        Labeled compartment mask.
        Example:
            0 = background
            1 = main compartment
            2 = epiphysis compartment

    whole_bone_mask : np.ndarray
        Binary full bone mask used only to calculate total bone length.

    label_to_use : int
        Label value in compartment_mask to extract from.
        Example: 1 for main compartment.

    percent_length : float
        Region length as percent of full bone length.

    axis : int
        Bone long axis. In your notebook, z is usually axis=2.

    start_from : str
        Which end of the selected label to start from:
        - "max" starts from the highest index along axis
        - "min" starts from the lowest index along axis

    Returns
    -------
    subregion_mask : np.ndarray
        Binary mask of the extracted subregion.

    info : dict
        Length and slice information.
    """

    compartment_mask = np.asarray(compartment_mask)
    whole_bone_mask = np.asarray(whole_bone_mask).astype(bool)

    if compartment_mask.shape != whole_bone_mask.shape:
        raise ValueError(
            f"compartment_mask and whole_bone_mask must have same shape, "
            f"got {compartment_mask.shape} and {whole_bone_mask.shape}"
        )

    selected_compartment = compartment_mask == label_to_use

    if not np.any(selected_compartment):
        raise ValueError(f"label_to_use={label_to_use} not found in compartment_mask.")

    # Full bone length
    bone_coords = np.argwhere(whole_bone_mask)

    if bone_coords.size == 0:
        raise ValueError("whole_bone_mask is empty.")

    bone_start = int(bone_coords[:, axis].min())
    bone_end = int(bone_coords[:, axis].max()) + 1
    bone_length_vox = bone_end - bone_start

    region_length_vox = int(round(bone_length_vox * percent_length / 100))
    region_length_vox = max(region_length_vox, 1)

    # Bounds of selected compartment label
    comp_coords = np.argwhere(selected_compartment)

    comp_start = int(comp_coords[:, axis].min())
    comp_end = int(comp_coords[:, axis].max()) + 1

    if start_from == "max":
        region_end = comp_end
        region_start = max(comp_start, comp_end - region_length_vox)
    elif start_from == "min":
        region_start = comp_start
        region_end = min(comp_end, comp_start + region_length_vox)
    else:
        raise ValueError("start_from must be 'max' or 'min'.")

    subregion_mask = np.zeros_like(selected_compartment, dtype=bool)

    slicer = [slice(None)] * 3
    slicer[axis] = slice(region_start, region_end)
    slicer = tuple(slicer)

    subregion_mask[slicer] = selected_compartment[slicer]

    info = {
        "label_to_use": label_to_use,
        "axis": axis,
        "start_from": start_from,
        "bone_start": bone_start,
        "bone_end": bone_end,
        "bone_length_vox": bone_length_vox,
        "percent_length": percent_length,
        "region_length_vox": region_length_vox,
        "compartment_start": comp_start,
        "compartment_end": comp_end,
        "region_start": region_start,
        "region_end": region_end,
        "subregion_voxels": int(np.count_nonzero(subregion_mask)),
    }

    return subregion_mask.astype(np.uint8), info


def detect_tibfib_junction_from_mask(
    whole_bone_mask,
    axis=2,
    smooth_sigma=3,
    min_second_component_area_vox=50,
    persistence_slices=5,
    search_fraction=(0.2, 0.9),
    mode="first_persistent",
):
    """
    Automatically estimate the tib-fib junction slice from a whole-bone mask.

    The method looks for the appearance of a persistent second 2D connected
    component in axial slices, which often corresponds to the fibula or
    tib-fib complex.

    Parameters
    ----------
    whole_bone_mask : np.ndarray
        Binary whole-bone mask.

    axis : int
        Longitudinal axis. In this notebook, z is usually axis=2.

    smooth_sigma : float
        Sigma for smoothing area profiles along the long axis.

    min_second_component_area_vox : int
        Minimum 2D area for the second-largest component to count as fibula.

    persistence_slices : int
        Number of consecutive slices required.

    search_fraction : tuple
        Fractional search range along the bone length, e.g. (0.2, 0.9).

    mode : str
        "first_persistent" finds the first persistent second component.
        "max_derivative" finds the strongest increase in second-component area.

    Returns
    -------
    junction_slice : int
        Estimated tib-fib junction slice index in full-resolution coordinates.

    info : dict
        Diagnostic profiles and parameters.
    """

    whole_bone_mask = np.asarray(whole_bone_mask).astype(bool)

    if axis != 2:
        # Move selected axis to z-like position for easier slicing
        mask = np.moveaxis(whole_bone_mask, axis, 2)
    else:
        mask = whole_bone_mask

    n_slices = mask.shape[2]

    total_area = np.zeros(n_slices, dtype=float)
    largest_area = np.zeros(n_slices, dtype=float)
    second_area = np.zeros(n_slices, dtype=float)
    n_components = np.zeros(n_slices, dtype=int)
    width_x = np.zeros(n_slices, dtype=float)
    width_y = np.zeros(n_slices, dtype=float)

    structure_2d = ndi.generate_binary_structure(2, 1)

    for z in range(n_slices):
        sl = mask[:, :, z]

        total_area[z] = np.count_nonzero(sl)

        if total_area[z] == 0:
            continue

        coords = np.argwhere(sl)
        width_x[z] = coords[:, 0].max() - coords[:, 0].min() + 1
        width_y[z] = coords[:, 1].max() - coords[:, 1].min() + 1

        labels, n = ndi.label(sl, structure=structure_2d)
        n_components[z] = n

        if n > 0:
            counts = np.bincount(labels.ravel())[1:]  # skip background
            counts_sorted = np.sort(counts)[::-1]

            largest_area[z] = counts_sorted[0]

            if len(counts_sorted) > 1:
                second_area[z] = counts_sorted[1]

    # Smooth profiles
    second_area_smooth = ndi.gaussian_filter1d(second_area, sigma=smooth_sigma)
    total_area_smooth = ndi.gaussian_filter1d(total_area, sigma=smooth_sigma)
    width_x_smooth = ndi.gaussian_filter1d(width_x, sigma=smooth_sigma)
    width_y_smooth = ndi.gaussian_filter1d(width_y, sigma=smooth_sigma)

    # Bone extent
    occupied = np.where(total_area > 0)[0]
    if occupied.size == 0:
        raise ValueError("whole_bone_mask is empty.")

    bone_start = int(occupied.min())
    bone_end = int(occupied.max()) + 1
    bone_length = bone_end - bone_start

    search_start = int(bone_start + search_fraction[0] * bone_length)
    search_end = int(bone_start + search_fraction[1] * bone_length)

    search_start = max(search_start, bone_start)
    search_end = min(search_end, bone_end)

    if mode == "first_persistent":
        candidate = second_area_smooth >= min_second_component_area_vox

        junction_slice = None

        for z in range(search_start, search_end - persistence_slices + 1):
            if np.all(candidate[z:z + persistence_slices]):
                junction_slice = z
                break

        if junction_slice is None:
            # fallback: largest derivative of second component area
            derivative = np.gradient(second_area_smooth)
            z_rel = np.argmax(derivative[search_start:search_end])
            junction_slice = search_start + int(z_rel)

    elif mode == "max_derivative":
        derivative = np.gradient(second_area_smooth)
        z_rel = np.argmax(derivative[search_start:search_end])
        junction_slice = search_start + int(z_rel)

    else:
        raise ValueError("mode must be 'first_persistent' or 'max_derivative'.")

    info = {
        "axis": axis,
        "junction_slice": int(junction_slice),
        "bone_start": bone_start,
        "bone_end": bone_end,
        "bone_length_vox": bone_length,
        "search_start": search_start,
        "search_end": search_end,
        "total_area": total_area,
        "largest_area": largest_area,
        "second_area": second_area,
        "second_area_smooth": second_area_smooth,
        "n_components": n_components,
        "width_x": width_x,
        "width_y": width_y,
        "total_area_smooth": total_area_smooth,
        "width_x_smooth": width_x_smooth,
        "width_y_smooth": width_y_smooth,
        "parameters": {
            "smooth_sigma": smooth_sigma,
            "min_second_component_area_vox": min_second_component_area_vox,
            "persistence_slices": persistence_slices,
            "search_fraction": search_fraction,
            "mode": mode,
        },
    }

    return int(junction_slice), info

def show_detected_slice_sagittal(
    image,
    detected_slice,
    mask=None,
    x=None,
    ds=1,
    title="Detected slice QC",
    cmap="gray",
    line_color="red",
):
    """
    Show a sagittal view with a line marking the detected axial slice.

    Parameters
    ----------
    image : np.ndarray
        3D image in x, y, z order.

    detected_slice : int
        Detected slice index along z / axis=2.

    mask : np.ndarray or None
        Optional mask overlay, same shape as image.

    x : int or None
        Sagittal x index to display. If None, uses mask/image center.

    ds : int
        Downsampling for display only.

    title : str
        Plot title.

    cmap : str
        Image colormap.

    line_color : str
        Colour of detected slice line.
    """

    image = np.asarray(image)

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != image.shape:
            raise ValueError("image and mask must have the same shape.")

    if x is None:
        if mask is not None and np.any(mask):
            coords = np.argwhere(mask)
            x = int(np.round(coords[:, 0].mean()))
        else:
            x = image.shape[0] // 2

    # Downsample for display
    image_disp = image[::ds, ::ds, ::ds]
    detected_slice_disp = int(round(detected_slice / ds))
    x_disp = int(round(x / ds))

    if mask is not None:
        mask_disp = mask[::ds, ::ds, ::ds]
    else:
        mask_disp = None

    vals = image_disp[np.isfinite(image_disp)]
    vmin = np.percentile(vals, 1)
    vmax = np.percentile(vals, 99.5)

    plt.figure(figsize=(6, 10))

    # Sagittal slice: y by z
    plt.imshow(
        image_disp[x_disp, :, :].T,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    if mask_disp is not None:
        plt.imshow(
            np.ma.masked_where(
                mask_disp[x_disp, :, :].T == 0,
                mask_disp[x_disp, :, :].T,
            ),
            origin="lower",
            alpha=0.30,
            interpolation="nearest",
        )

    # z is vertical axis after transpose
    plt.axhline(
        detected_slice_disp,
        color=line_color,
        linewidth=2,
        linestyle="--",
    )

    plt.title(
        f"{title}\n"
        f"x = {x}, detected z = {detected_slice}"
    )
    plt.xlabel("y")
    plt.ylabel("z")
    plt.axis("on")
    plt.tight_layout()
    plt.show()
	 
	 
def threshold_fat_mad(
    image,
    mask,
    gaussian_sigma_vox=1.0,
    k=3.5,
):
    """
    Segment bright fat signal using:
        threshold = median + k * 1.4826 * MAD
    """
    image = np.asarray(image).astype(np.float32)
    mask = np.asarray(mask).astype(bool)

    if gaussian_sigma_vox and gaussian_sigma_vox > 0:
        image_smooth = ndi.gaussian_filter(image, sigma=gaussian_sigma_vox)
    else:
        image_smooth = image

    values = image_smooth[mask & np.isfinite(image_smooth)]
    values = values[values != 0]

    med = np.median(values)
    mad = np.median(np.abs(values - med))
    mad_sigma = 1.4826 * mad

    threshold = med + k * mad_sigma

    fat_mask = (image_smooth >= threshold) & mask

    return fat_mask.astype(np.uint8), float(threshold), image_smooth

def extract_region_from_slice(
    compartment_mask,
    start_slice,
    axis=2,
    direction="distal_max",
):
    """
    Extract region of a compartment mask extending from a selected slice.

    Parameters
    ----------
    compartment_mask : np.ndarray
        Binary compartment mask to crop.

    start_slice : int
        Slice index defining the tib-fib junction.

    axis : int
        Longitudinal axis. In this notebook, z is usually axis=2.

    direction : str
        - "distal_max": keep slices from start_slice to max end
        - "distal_min": keep slices from min end to start_slice

    Returns
    -------
    region_mask : np.ndarray
        Binary subregion mask.

    info : dict
        Region slice information.
    """

    compartment_mask = np.asarray(compartment_mask).astype(bool)

    region_mask = np.zeros_like(compartment_mask, dtype=bool)

    if direction == "distal_max":
        region_slice = slice(start_slice, compartment_mask.shape[axis])
    elif direction == "distal_min":
        region_slice = slice(0, start_slice + 1)
    else:
        raise ValueError("direction must be 'distal_max' or 'distal_min'.")

    slicer = [slice(None)] * 3
    slicer[axis] = region_slice
    slicer = tuple(slicer)

    region_mask[slicer] = compartment_mask[slicer]

    info = {
        "axis": axis,
        "start_slice": int(start_slice),
        "direction": direction,
        "region_voxels": int(np.count_nonzero(region_mask)),
    }

    return region_mask.astype(np.uint8), info


def tibfib_slice_picker(
    image,
    mask=None,
    ds=2,
    title="Select tib-fib junction slice",
    cmap="gray",
):
    """
    Axial viewer to manually select the tib-fib junction slice.

    Parameters
    ----------
    image : np.ndarray
        3D image in x, y, z order.

    mask : np.ndarray or None
        Optional mask overlay to help identify the slice.

    ds : int
        Downsampling factor for display.

    Returns
    -------
    viewer : dict
        Contains the z slider. Selected full-resolution slice is:
            viewer["z_slider"].value * ds
    """

    image = np.asarray(image)

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != image.shape:
            raise ValueError("image and mask must have the same shape.")

    image_disp = image[::ds, ::ds, ::ds]
    mask_disp = mask[::ds, ::ds, ::ds] if mask is not None else None

    values = image_disp[np.isfinite(image_disp)]
    vmin = np.percentile(values, 1)
    vmax = np.percentile(values, 99.5)

    if mask_disp is not None and np.any(mask_disp):
        coords = np.argwhere(mask_disp)
        z0 = int(np.round(coords[:, 2].mean()))
    else:
        z0 = image_disp.shape[2] // 2

    s_z = widgets.IntSlider(
        value=z0,
        min=0,
        max=image_disp.shape[2] - 1,
        step=1,
        description="Axial z",
        continuous_update=False,
    )

    out = widgets.Output()

    with plt.ioff():
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        image_disp[:, :, s_z.value].T,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    if mask_disp is not None:
        ov = ax.imshow(
            np.ma.masked_where(
                mask_disp[:, :, s_z.value].T == 0,
                mask_disp[:, :, s_z.value].T,
            ),
            cmap=ListedColormap(["red"]),
            origin="lower",
            alpha=0.35,
            interpolation="nearest",
        )
    else:
        ov = None

    ax.axis("off")

    def update(_=None):
        z = s_z.value
        im.set_data(image_disp[:, :, z].T)

        if ov is not None:
            ov.set_data(
                np.ma.masked_where(
                    mask_disp[:, :, z].T == 0,
                    mask_disp[:, :, z].T,
                )
            )

        ax.set_title(
            f"{title}\n"
            f"display z = {z}, full-res z = {z * ds}"
        )

        with out:
            clear_output(wait=True)
            print(f"Selected tib-fib junction slice: {z * ds}")

        fig.canvas.draw_idle()

    s_z.observe(update, names="value")

    update()
    display(VBox([s_z, fig.canvas, out]))

    return {
        "z_slider": s_z,
        "ds": ds,
        "figure": fig,
        "axis": ax,
    }

def extract_region_from_slice(
    compartment_mask,
    start_slice,
    axis=2,
    direction="distal_max",
):
    """
    Extract region of a compartment mask extending from a selected slice.

    Parameters
    ----------
    compartment_mask : np.ndarray
        Binary compartment mask to crop.

    start_slice : int
        Slice index defining the tib-fib junction.

    axis : int
        Longitudinal axis. In this notebook, z is usually axis=2.

    direction : str
        - "distal_max": keep slices from start_slice to max end
        - "distal_min": keep slices from min end to start_slice

    Returns
    -------
    region_mask : np.ndarray
        Binary subregion mask.

    info : dict
        Region slice information.
    """

    compartment_mask = np.asarray(compartment_mask).astype(bool)

    region_mask = np.zeros_like(compartment_mask, dtype=bool)

    if direction == "distal_max":
        region_slice = slice(start_slice, compartment_mask.shape[axis])
    elif direction == "distal_min":
        region_slice = slice(0, start_slice + 1)
    else:
        raise ValueError("direction must be 'distal_max' or 'distal_min'.")

    slicer = [slice(None)] * 3
    slicer[axis] = region_slice
    slicer = tuple(slicer)

    region_mask[slicer] = compartment_mask[slicer]

    info = {
        "axis": axis,
        "start_slice": int(start_slice),
        "direction": direction,
        "region_voxels": int(np.count_nonzero(region_mask)),
    }

    return region_mask.astype(np.uint8), info


def label_mask_regions(
    masks,
    labels=None,
    priority_order=None,
    dtype=np.uint8,
    verbose=True,
):
    """
    Combine multiple binary masks into one labeled image.

    Parameters
    ----------
    masks : dict
        Dictionary of binary masks:
            {
                "region name": mask_array,
                ...
            }

    labels : dict or None
        Dictionary assigning integer labels to each mask:
            {
                "region name": label_value,
                ...
            }
        If None, labels are assigned as 1, 2, 3, ...

    priority_order : list or None
        Order in which labels are assigned.
        Later regions overwrite earlier regions if masks overlap.
        If None, uses the insertion order of masks.

    dtype : numpy dtype
        Output dtype. np.uint8 is fine for labels 0–255.

    verbose : bool
        If True, print label summary.

    Returns
    -------
    labeled_mask : np.ndarray
        Labeled image.

    info : dict
        Dictionary containing label map and voxel counts.
    """

    if not isinstance(masks, dict):
        raise ValueError("masks must be a dictionary: {'name': mask_array}")

    if len(masks) == 0:
        raise ValueError("masks dictionary is empty")

    # Use insertion order unless otherwise specified
    if priority_order is None:
        priority_order = list(masks.keys())

    # Check all names exist
    for name in priority_order:
        if name not in masks:
            raise ValueError(f"'{name}' is in priority_order but not in masks")

    # Check shapes
    first_name = priority_order[0]
    reference_shape = np.asarray(masks[first_name]).shape

    for name, mask in masks.items():
        if np.asarray(mask).shape != reference_shape:
            raise ValueError(
                f"Mask '{name}' has shape {np.asarray(mask).shape}, "
                f"but expected {reference_shape}"
            )

    # Assign default labels if not provided
    if labels is None:
        labels = {
            name: i + 1
            for i, name in enumerate(priority_order)
        }

    # Check labels for all masks
    for name in priority_order:
        if name not in labels:
            raise ValueError(f"No label value provided for mask '{name}'")

    labeled_mask = np.zeros(reference_shape, dtype=dtype)

    # Later masks overwrite earlier masks if there is overlap
    for name in priority_order:
        region = np.asarray(masks[name]).astype(bool)
        labeled_mask[region] = labels[name]

    # Summary
    label_counts = {
        "background": int(np.count_nonzero(labeled_mask == 0))
    }

    for name in priority_order:
        label_value = labels[name]
        label_counts[name] = int(np.count_nonzero(labeled_mask == label_value))

    info = {
        "labels": labels,
        "priority_order": priority_order,
        "label_counts": label_counts,
        "unique_values": np.unique(labeled_mask).tolist(),
    }

    if verbose:
        print("Unique labels:", info["unique_values"])
        for name, count in label_counts.items():
            if name == "background":
                print(f"{name}: {count}")
            else:
                print(f"{name} label {labels[name]}: {count}")

    return labeled_mask, info