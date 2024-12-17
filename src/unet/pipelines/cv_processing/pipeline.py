from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from unet.utils.dataset import contour_process_cv
from .nodes import (
    img_process,
    create_cv_data_dict,
    prepare_cv_dataset,
    run_processing,
)

def create_pipeline(**kwargs) -> Pipeline:
    # Base pipeline that uses the processor and model
    # base_pipeline = Pipeline(
    #     [
    #         node(
    #             func=img_process,
    #             inputs=["raw_data", "cv_processing_config", "roi"],
    #             outputs="cv_processed_set",
    #             name="cv_process_set"
    #         ),
    #         node(
    #             func=contour_process_cv,
    #              inputs=["cv_processed_set"],
    #              outputs="cv_processed_contours",
    #              name="contour_process_cv"
    #         )
    #     ]
    # )
    base_pipeline = Pipeline(
        [
            node(
                func=create_cv_data_dict,
                inputs=["raw_data", "roi"],
                outputs="cv_data_dict",
                name="create_cv_data_dict"
            ),
            node(
                func=prepare_cv_dataset,
                inputs=["cv_data_dict", "cv_processing_config"],
                outputs="cv_dataset",
                name="prepare_cv_dataset"
            ),
            node(
                func=run_processing,
                inputs=["cv_dataset", "cv_processing_config"],
                outputs="cv_processed",
                name="run_cv_processing"
            ),
            node(
                func=contour_process_cv,
                inputs="cv_processed",
                outputs="cv_processed_contours",
                name="cv_contour_process"
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
