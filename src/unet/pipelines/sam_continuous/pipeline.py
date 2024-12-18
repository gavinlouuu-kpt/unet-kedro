"""
This is a boilerplate pipeline 'sam_continuous'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline
from unet.utils.dataset import filter_empty_frames_ext_bg, select_roi, select_background
from unet.pipelines.sam_inference.pipeline import get_sam_inference_pipeline
from unet.pipelines.sam_inference.nodes import initSAM

def create_pipeline(**kwargs) -> Pipeline:
    init_sam_nodes = Pipeline(
        [
            node(
                func=initSAM,
                inputs=["params:sam_bead_inference"],
                outputs=["processor", "model"],
                name="initSAM"
            ),
        ]
    )

    background_init_pipeline = pipeline(
        [
            node(
                func=select_roi,
                inputs=["continuous_partition"],
                outputs="roi",
                name="select_roi",
            ),
            node(
                func=select_background,
                inputs=["continuous_partition", "roi"],
                outputs="background",
                name="select_background",
            ),
            node(
                func=filter_empty_frames_ext_bg,
                inputs=["continuous_partition", "params:cv_basic_options", "roi", "background"],
                outputs="raw_data",
                name="filter_empty_frames_ext_bg",
            ),
        ]
    )

    # Create the complete pipeline with namespace
    complete_pipeline = pipeline(
        pipe=background_init_pipeline + get_sam_inference_pipeline(),
        namespace="paa6",
        inputs={
            # "continuous_partition": "paa6.continuous_partition",
            "processor": "processor",
            "model": "model",
        },
        parameters={"params:cv_basic_options": "params:cv_basic_options"},
        # outputs={"processed_predictions": "paa6.processed_predictions"}
    )
    
    return init_sam_nodes + complete_pipeline


