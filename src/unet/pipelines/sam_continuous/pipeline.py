"""
This is a boilerplate pipeline 'sam_continuous'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline
from unet.utils.dataset import filter_empty_frames_ext_bg, select_roi, select_background
from unet.pipelines.sam_inference.pipeline import sam_inference_pipeline

def create_pipeline(**kwargs) -> Pipeline:

    sam_inference_base = sam_inference_pipeline()
    
    sam_inference_pipeline = pipeline(
        pipe=sam_inference_base,
        namespace="paa6",
        parameters={"params:sam_bead_inference": "params:sam_bead_inference"},
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
                outputs="filtered_images",
                name="filter_empty_frames",
            ),
        ]
    )
    paa6_pipeline = pipeline(
        pipe=background_init_pipeline,
        namespace="paa6",
        # inputs={"": ""},
        parameters={"params:cv_basic_options": "params:cv_basic_options"},
    )
    return paa6_pipeline


