from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (
    img_process
)

def create_pipeline(**kwargs) -> Pipeline:
    # Base pipeline that uses the processor and model
    base_pipeline = Pipeline(
        [
            node(
                func=img_process,
                inputs=["raw_data", "cv_processing_config", "roi"],
                outputs="cv_processed_set",
                name="cv_process_set"
            )
        ]
    )

    # Create pipelines for each namespace
    PAA_12_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="PAA_12",
    )

    PAA_10_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="PAA_10",
    )

    PAA_8_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="PAA_8",
    )

    # Combine all pipelines
    return PAA_12_pipeline + PAA_10_pipeline + PAA_8_pipeline