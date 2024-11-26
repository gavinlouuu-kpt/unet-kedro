"""
This is a boilerplate pipeline 'sam_predictor'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import initialize_sam_base, prepare_masks, prepare_training_data, train_sam

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=initialize_sam_base,
            inputs=["params:tune_sam"],
            outputs=["sam", "optimizer"],
            name="initialize_sam_base",
        ),
        node(
            func=prepare_masks,
            inputs=["paa_12_label"],
            outputs="masks",
            name="prepare_masks",
        ),
        node(
            func=prepare_training_data,
            inputs=["paa_12_image", "masks", "params:tune_sam", "paa_12_roi"],
            outputs="training_data",
            name="prepare_training_data",
        ),
        node(
            func=train_sam,
            inputs=["sam", "optimizer", "training_data", "params:tune_sam"],
            outputs=["trained_sam", "training_metrics"],
            name="train_sam",
        ),
    ])
