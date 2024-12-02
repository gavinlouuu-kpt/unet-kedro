# from kedro.pipeline import Pipeline, node
# from kedro.pipeline.modular_pipeline import pipeline
# from .nodes import (
#     create_data_dict,
#     initSAM,
#     run_inference,
#     prepare_dataloader,
#     run_inference,
#     processingDI
# )

# base_pipeline = pipeline(
#         [
#             node(
#                 func=create_data_dict, 
#                 inputs=["raw_data","roi"], 
#                 outputs="data_dict",
#                 name="create_data_dict"
#             ),
#             node(
#                 func=prepare_dataloader, 
#                 inputs=["data_dict","processor"], 
#                 outputs="dataloader", 
#                 name="prepare_dataloader"
#             ),
#             node(
#                 func=run_inference, 
#                 inputs=["dataloader","model"], 
#                 outputs="predictions", 
#                 name="run_inference"
#             ),
#             node(
#                 func=processingDI, 
#                 inputs=["predictions"], 
#                 outputs="processed_predictions",
#                 name="processingDI"
#             ),
#         ]
#     )

# def create_pipeline(**kwargs) -> Pipeline:
    
    
#     common_nodes = pipeline(
#         [
#             node(
#                 func=initSAM, 
#                 inputs=["params:sam_bead_inference"], 
#                 outputs=["processor","model"],
#                 name="initSAM"
#             ),
#         ],
#     )

#     PAA_12_pipeline = pipeline(
#         pipe=base_pipeline,
#         namespace="PAA_12",
#     )

#     PAA_10_pipeline = pipeline(
#         pipe=base_pipeline,
#         namespace="PAA_10",
#     )

#     PAA_8_pipeline = pipeline(
#         pipe=base_pipeline,
#         namespace="PAA_8",
#     )
    
#     return common_nodes + PAA_12_pipeline + PAA_10_pipeline + PAA_8_pipeline

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (
    create_data_dict,
    initSAM,
    run_inference,
    prepare_dataloader,
    processingDI
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
    base_pipeline = Pipeline(
        [
            node(
                func=create_data_dict,
                inputs=["raw_data", "roi"],
                outputs="data_dict",
                name="create_data_dict"
            ),
            node(
                func=prepare_dataloader,
                inputs=["data_dict", "processor"],  # Use global processor
                outputs="dataloader",
                name="prepare_dataloader"
            ),
            node(
                func=run_inference,
                inputs=["dataloader", "model"],  # Use global model
                outputs="predictions",
                name="run_inference"
            ),
            node(
                func=processingDI,
                inputs=["predictions"],
                outputs="processed_predictions",
                name="processingDI"
            ),
        ]
    )

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

    # Combine all pipelines
    return common_nodes + PAA_12_pipeline + PAA_10_pipeline + PAA_8_pipeline