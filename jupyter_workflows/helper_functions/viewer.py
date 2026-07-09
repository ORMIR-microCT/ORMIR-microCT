import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.colors import ListedColormap
from IPython.display import display
from ipywidgets import VBox
import ipywidgets as widgets

def show_multi_mask_overlay_viewer(
    image,
    masks,
    colors=None,
    ds=2,
    alpha=0.45,
    cmap="gray",
    continuous_update=False,
    figsize=(12, 5),
    title_prefix="Mask overlay",
):
    """
    Interactive 3-plane viewer for a 3D image with multiple binary mask overlays.

    Parameters
    ----------
    image : np.ndarray
        3D greyscale image.

    masks : dict
        Dictionary of binary masks:
            {
                "mask name": mask_array,
                ...
            }

    colors : dict or None
        Dictionary of colours matching mask names:
            {
                "mask name": "red",
                ...
            }
        If None, default colours are assigned.

    ds : int
        Downsampling factor for display only.

    alpha : float
        Overlay transparency.

    cmap : str
        Greyscale image colormap.

    continuous_update : bool
        Whether sliders update continuously while dragging.

    figsize : tuple
        Figure size.

    title_prefix : str
        Figure title prefix.
    """

    image = np.asarray(image)

    if not isinstance(masks, dict):
        raise ValueError("masks must be a dictionary: {'name': mask_array}")

    if len(masks) == 0:
        raise ValueError("masks dictionary is empty")

    if ds < 1:
        raise ValueError("ds must be >= 1")

    # Default colours if none supplied
    default_colors = [
        "red",
        "cyan",
        "yellow",
        "lime",
        "magenta",
        "orange",
        "blue",
        "white",
    ]

    if colors is None:
        colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(masks.keys())
        }

    # Validate and downsample masks
    masks_bool = {}
    masks_disp = {}

    for name, mask in masks.items():
        mask = np.asarray(mask) > 0

        if mask.shape != image.shape:
            raise ValueError(
                f"Mask '{name}' has shape {mask.shape}, but image has shape {image.shape}"
            )

        masks_bool[name] = mask
        masks_disp[name] = mask[::ds, ::ds, ::ds]

        if name not in colors:
            colors[name] = default_colors[len(colors) % len(default_colors)]

    image_disp = image[::ds, ::ds, ::ds]

    # Robust display range
    vmin = np.percentile(image_disp, 1)
    vmax = np.percentile(image_disp, 99.9)

    # Find center of union mask
    union_mask = np.zeros_like(next(iter(masks_disp.values())), dtype=bool)
    for m in masks_disp.values():
        union_mask |= m

    coords = np.argwhere(union_mask > 0)
    if coords.size == 0:
        center = np.array(image_disp.shape) // 2
    else:
        center = np.round(coords.mean(axis=0)).astype(int)

    def masked_slice(mask_slice):
        return np.ma.masked_where(mask_slice == 0, mask_slice)

    s_x = widgets.IntSlider(
        value=int(center[0]),
        min=0,
        max=image_disp.shape[0] - 1,
        step=1,
        description="Sagittal x",
        continuous_update=continuous_update,
    )

    s_y = widgets.IntSlider(
        value=int(center[1]),
        min=0,
        max=image_disp.shape[1] - 1,
        step=1,
        description="Coronal y",
        continuous_update=continuous_update,
    )

    s_z = widgets.IntSlider(
        value=int(center[2]),
        min=0,
        max=image_disp.shape[2] - 1,
        step=1,
        description="Axial z",
        continuous_update=continuous_update,
    )

    with plt.ioff():
        fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Base images
    im_x = axes[0].imshow(
        image_disp[s_x.value, :, :].T,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    im_y = axes[1].imshow(
        image_disp[:, s_y.value, :].T,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    im_z = axes[2].imshow(
        image_disp[:, :, s_z.value].T,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Overlay masks
    overlay_artists = {"x": {}, "y": {}, "z": {}}

    for name, mask_disp in masks_disp.items():
        mask_cmap = ListedColormap([colors[name]])

        overlay_artists["x"][name] = axes[0].imshow(
            masked_slice(mask_disp[s_x.value, :, :].T),
            cmap=mask_cmap,
            origin="lower",
            alpha=alpha,
            interpolation="nearest",
        )

        overlay_artists["y"][name] = axes[1].imshow(
            masked_slice(mask_disp[:, s_y.value, :].T),
            cmap=mask_cmap,
            origin="lower",
            alpha=alpha,
            interpolation="nearest",
        )

        overlay_artists["z"][name] = axes[2].imshow(
            masked_slice(mask_disp[:, :, s_z.value].T),
            cmap=mask_cmap,
            origin="lower",
            alpha=alpha,
            interpolation="nearest",
        )

    axes[0].set_title(f"{title_prefix}\nSagittal x = {s_x.value * ds}")
    axes[1].set_title(f"{title_prefix}\nCoronal y = {s_y.value * ds}")
    axes[2].set_title(f"{title_prefix}\nAxial z = {s_z.value * ds}")

    for ax in axes:
        ax.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors[name], label=name)
        for name in masks.keys()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(len(legend_patches), 4),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    def update_overlay(_=None):
        x = s_x.value
        y = s_y.value
        z = s_z.value

        im_x.set_data(image_disp[x, :, :].T)
        im_y.set_data(image_disp[:, y, :].T)
        im_z.set_data(image_disp[:, :, z].T)

        for name, mask_disp in masks_disp.items():
            overlay_artists["x"][name].set_data(
                masked_slice(mask_disp[x, :, :].T)
            )
            overlay_artists["y"][name].set_data(
                masked_slice(mask_disp[:, y, :].T)
            )
            overlay_artists["z"][name].set_data(
                masked_slice(mask_disp[:, :, z].T)
            )

        axes[0].set_title(f"{title_prefix}\nSagittal x = {x * ds}")
        axes[1].set_title(f"{title_prefix}\nCoronal y = {y * ds}")
        axes[2].set_title(f"{title_prefix}\nAxial z = {z * ds}")

        fig.canvas.draw_idle()

    s_x.observe(update_overlay, names="value")
    s_y.observe(update_overlay, names="value")
    s_z.observe(update_overlay, names="value")

    display(VBox([VBox([s_x, s_y, s_z]), fig.canvas]))

    update_overlay()

    return {
        "figure": fig,
        "axes": axes,
        "sliders": {"x": s_x, "y": s_y, "z": s_z},
        "image_display": image_disp,
        "masks_display": masks_disp,
    }

def qc_masks(image, masks, colors=None, ds=2, alpha=0.45, title="QC masks"):
    show_multi_mask_overlay_viewer(
        image=image,
        masks=masks,
        colors=colors,
        ds=ds,
        alpha=alpha,
        title_prefix=title,
    )
    return None

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

    plt.figure(figsize=(3, 5))

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