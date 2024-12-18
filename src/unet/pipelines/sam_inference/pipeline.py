from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from unet.utils.dataset import contour_process_cv
from .nodes import (
    create_data_dict,
    initSAM,
    run_inference,
    prepare_dataloader,
)

def get_sam_inference_pipeline() -> Pipeline:
    """Returns the base pipeline for SAM inference."""
    return Pipeline(
        [
            node(
                func=create_data_dict,
                inputs=["raw_data", "roi"],
                outputs="data_dict",
                name="create_data_dict"
            ),
            node(
                func=prepare_dataloader,
                inputs=["data_dict", "processor"],
                outputs="dataloader",
                name="prepare_dataloader"
            ),
            node(
                func=run_inference,
                inputs=["dataloader", "model"],
                outputs="predictions",
                name="run_inference"
            ),
            node(
                func=contour_process_cv,
                inputs=["predictions"],
                outputs="processed_predictions",
                name="sam_contour_process"
            ),
        ]
    )

def create_pipeline(**kwargs) -> Pipeline:
    # Initialize processor and model once
    common_nodes = Pipeline(
        [
            node(
                func=initSAM,
                inputs=["params:sam_bead_inference"],
                outputs=["processor", "model"],
                name="initSAM"
            ),
        ]
    )

    # Base pipeline that uses the processor and model
    base_pipeline = get_sam_inference_pipeline()
    # Create pipelines for each namespace
    PAA_12_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="PAA_12",
        inputs={
            "raw_data": "PAA_12.raw_data",
            "roi": "PAA_12.roi",
            "processor": "processor",
            "model": "model"
        },
        outputs={"processed_predictions": "PAA_12.processed_predictions"}
    )

    PAA_10_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="PAA_10",
        inputs={
            "raw_data": "PAA_10.raw_data",
            "roi": "PAA_10.roi",
            "processor": "processor",
            "model": "model"
        },
        outputs={"processed_predictions": "PAA_10.processed_predictions"}
    )

    PAA_8_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="PAA_8",
        inputs={
            "raw_data": "PAA_8.raw_data",
            "roi": "PAA_8.roi",
            "processor": "processor",
            "model": "model"
        },
        outputs={"processed_predictions": "PAA_8.processed_predictions"}
    )

    G1_20241205_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="G1_20241205",
        inputs={
            "raw_data": "G1_20241205.raw_data",
            "roi": "G1_20241205.roi",
            "processor": "processor",
            "model": "model"
        },
        outputs={"processed_predictions": "G1_20241205.processed_predictions"}
    )

    G2_20241205_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="G2_20241205",
        inputs={
            "raw_data": "G2_20241205.raw_data",
            "roi": "G2_20241205.roi",
            "processor": "processor",
            "model": "model"
        },
        outputs={"processed_predictions": "G2_20241205.processed_predictions"}
    )

    G3_20241205_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="G3_20241205",
        inputs={
            "raw_data": "G3_20241205.raw_data",
            "roi": "G3_20241205.roi",
            "processor": "processor",
            "model": "model"
        },
        outputs={"processed_predictions": "G3_20241205.processed_predictions"}
    )

    G4_20241205_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="G4_20241205",
        inputs={
            "raw_data": "G4_20241205.raw_data",
            "roi": "G4_20241205.roi",
            "processor": "processor",
            "model": "model"
        },
        outputs={"processed_predictions": "G4_20241205.processed_predictions"}
    )

    test_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="test",
        inputs={
            "raw_data": "test.raw_data",
            "roi": "test.roi",
            "processor": "processor",
            "model": "model"
        },
        outputs={"processed_predictions": "test.processed_predictions"}
    )

    # Combine all pipelines
    # return common_nodes + PAA_12_pipeline + PAA_10_pipeline + PAA_8_pipeline
    return common_nodes + G1_20241205_pipeline + G2_20241205_pipeline + G3_20241205_pipeline + G4_20241205_pipeline
    # return common_nodes + test_pipeline