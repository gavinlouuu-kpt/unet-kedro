from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from unet.utils.dataset import load_partition_dict
from .nodes import (
    create_mask_overlays,
    create_scatter_plots_with_csv,
    create_interactive_scatter_plots,
)

def create_pipeline(**kwargs) -> Pipeline:

    base_pipeline = Pipeline(
        [
            # node(
            #     func=create_mask_overlays,
            #     inputs="collection",
            #     outputs="mask_overlays",
            #     name="create_sam_mask_overlays"
            # ),
            node(
                func=create_scatter_plots_with_csv,
                inputs="collection",
                outputs=["scatter_plots", "DI_pts", "combined_scatter_plot"],
                name="create_scatter_plots_with_csv"
            ),
            node(
                func=create_interactive_scatter_plots,
                inputs="collection",
                outputs="interactive_scatter_plots",
                name="create_interactive_scatter_plots"
            )
        ]
    )

    # Create pipelines for each namespace
    SAM_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="SAM",
    )

    CV_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="CV",
    )


    # Combine all pipelines
    return SAM_pipeline + CV_pipeline
    # Combine all pipelines
    # return CV_pipeline