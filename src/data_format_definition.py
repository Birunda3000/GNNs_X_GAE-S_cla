from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Any


class Metadata(BaseModel):
    dataset_name: str
    feature_type: str
    num_nodes: int
    num_edges: int
    num_total_features: int
    processed_at: str
    directed: bool


class GraphStructure(BaseModel):
    edge_index: List[List[int]]
    y: List[Optional[int]]
    node_names: List[Optional[str]]

    @validator("edge_index")
    def check_edge_index_shape(cls, v: List[List[int]]) -> List[List[int]]:
        if len(v) != 2:
            raise ValueError("edge_index must have a shape of [2, num_edges]")
        if len(v[0]) != len(v[1]):
            raise ValueError(
                "Source and destination edge lists must have the same length"
            )
        return v


class NodeFeaturesEntry(BaseModel):
    indices: List[int]
    weights: List[float]

    @validator("weights")
    def check_weights_len(cls, v: List[float], values: Dict[str, Any]) -> List[float]:
        if "indices" in values and len(v) != len(values["indices"]):
            raise ValueError("indices and weights must have the same length")
        return v


class WSG(BaseModel):
    metadata: Metadata
    graph_structure: GraphStructure
    node_features: Dict[str, NodeFeaturesEntry]

    @validator("graph_structure")
    def check_graph_consistency(
        cls, v: GraphStructure, values: Dict[str, Any]
    ) -> GraphStructure:
        if "metadata" not in values:
            return v  # Metadata not validated yet, skip

        metadata: Metadata = values["metadata"]

        # Check edge counts
        if len(v.edge_index[0]) != metadata.num_edges:
            raise ValueError(
                f"Number of edges in edge_index ({len(v.edge_index[0])}) does not match metadata.num_edges ({metadata.num_edges})"
            )

        # Check node counts
        if len(v.y) != metadata.num_nodes:
            raise ValueError(
                f"Length of y ({len(v.y)}) does not match metadata.num_nodes ({metadata.num_nodes})"
            )
        if len(v.node_names) != metadata.num_nodes:
            raise ValueError(
                f"Length of node_names ({len(v.node_names)}) does not match metadata.num_nodes ({metadata.num_nodes})"
            )

        return v

    @validator("node_features")
    def check_node_and_feature_ids(
        cls, v: Dict[str, NodeFeaturesEntry], values: Dict[str, Any]
    ) -> Dict[str, NodeFeaturesEntry]:
        if "metadata" not in values:
            return v  # Metadata not validated yet, skip

        metadata: Metadata = values["metadata"]
        num_nodes = metadata.num_nodes
        num_features = metadata.num_total_features

        for node_id_str, feature_entry in v.items():
            node_id = int(node_id_str)
            if not (0 <= node_id < num_nodes):
                raise ValueError(
                    f"Node ID '{node_id}' in node_features is out of bounds [0, {num_nodes-1}]"
                )

            for feature_idx in feature_entry.indices:
                if not (0 <= feature_idx < num_features):
                    raise ValueError(
                        f"Feature index '{feature_idx}' for node '{node_id}' is out of bounds [0, {num_features-1}]"
                    )

        # Check edge index bounds
        if "graph_structure" in values:
            edge_index = values["graph_structure"].edge_index
            max_node_in_edges = max(max(edge_index[0]), max(edge_index[1]))
            if not (0 <= max_node_in_edges < num_nodes):
                raise ValueError(
                    f"Node ID '{max_node_in_edges}' in edge_index is out of bounds [0, {num_nodes-1}]"
                )

        return v
