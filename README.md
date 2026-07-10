# microCT Workflows

A collection of workflows for processing, analyzing, and segmenting microCT datasets.

## Repository Structure

### IPL_workflows

Workflows implemented using IPL (Image Processing Language).

### XamFlow_workflows

Workflows developed using XamFlow pipelines.

### jupyter_workflows

Jupyter Notebook-based workflows for interactive microCT image analysis.

#### Available Notebooks

##### mouse_tibia_mask_and_segment

This notebook provides a basic workflow for segmenting the bone marrow and epiphyseal compartments of a mouse tibia microCT scan.

**Assumptions:**

- The input file is a grayscale `.aim` image of a mouse tibia aligned axially along the Z-axis.
- An accompanying `mask.aim` file containing a mask of the whole bone is available.

> **Note:** This workflow is a work in progress and is based on several protocols described in the literature. Additional tuning and validation may still be required.

## Contributing

Contributions, improvements, and additional workflows are welcome. Feel free to submit a pull request.

## License

Please add the appropriate license information for your project.
