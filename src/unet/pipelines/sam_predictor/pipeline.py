"""
This is a boilerplate pipeline 'sam_predictor'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import initialize_sam

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=initialize_sam,
            inputs=["params:sam"],
            outputs="sam_predictor",
            name="initialize_sam",
        ),
    ])
