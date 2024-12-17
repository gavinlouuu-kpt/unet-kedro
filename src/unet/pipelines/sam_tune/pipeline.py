"""
This is a boilerplate pipeline 'sam_tune'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (
    init_processor,
    prepare_masks,
    prepare_training_data,
    combine_data_collections,
    create_dataloaders,
    create_fine_tuned_model
)


def create_pipeline(**kwargs) -> Pipeline:
    # Initialize processor
    init_processor_pipeline = Pipeline(
        [
            node(
                func=init_processor,
                inputs=["params:sam_fine_tune"],
                outputs="base_processor",
                name="init_processor"
            ),
        ]
    )

    model_training_pipeline = Pipeline(
        [
            node(
                func=combine_data_collections,
                inputs="datasets_collection",
                outputs="combined_dataset",
                name="combine_data_collections"
            ),
            node(
                func=create_dataloaders,
                inputs=["combined_dataset", "base_processor", "params:sam_fine_tune"],
                outputs=["train_loader", "val_loader"],
                name="create_dataloaders"
            ),
            node(
                func=create_fine_tuned_model,
                inputs=["train_loader", "val_loader", "base_processor", "params:sam_fine_tune"],
                outputs="fine_tuned_model",
                name="create_fine_tuned_model"
            ),
        ]
    )

    # Base pipeline that creates the data set
    base_pipeline = Pipeline(
        [
            node(
                func=prepare_masks,
                inputs=["sam_fine_tuning_mask"],
                outputs="masks",
                name="prepare_masks"
            ),
            node(
                func=prepare_training_data,
                inputs=["sam_fine_tuning_image", "masks", "params:sam_fine_tune", "tuning_roi"],
                outputs="part_datasets",
                name="prepare_training_data"
            ),
            
        ]
    )

    # Create pipelines for each namespace
    PAA_12_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="PAA_12",
        # inputs={
            # "base_processor": "base_processor",
        #     "sam_fine_tuning_mask": "PAA_12.sam_fine_tuning_mask",
        #     "roi": "PAA_12.roi",
        # },
        parameters={"params:sam_fine_tune": "params:sam_fine_tune"},
        # outputs={"prepared_training_data": "PAA_12.prepared_training_data"}
    )

    PAA_122_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="PAA_122",
        # inputs={
        #     "base_processor": "base_processor",
            # "sam_fine_tuning_image": "PAA_122.sam_fine_tuning_image",
            # "sam_fine_tuning_mask": "PAA_122.sam_fine_tuning_mask",
            # "roi": "PAA_122.roi",
        # },
        parameters={"params:sam_fine_tune": "params:sam_fine_tune"},
        # outputs={"prepared_training_data": "PAA_122.prepared_training_data"}
    )
    return init_processor_pipeline + PAA_12_pipeline + PAA_122_pipeline + model_training_pipeline
