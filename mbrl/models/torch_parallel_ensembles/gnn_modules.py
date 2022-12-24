from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from mbrl.models.utils import (
    unsorted_segment_mean_ensemble,
    unsorted_segment_sum_ensemble,
)

from .simple import MLPParallelEnsemble

TensorType = Union[torch.Tensor, np.ndarray]


class GraphNeuralNetworkEnsemble(nn.Module):
    """A graph neural network implementation for the dynamics model.
    Code taken and modified from: https://github.com/tkipf/c-swm
    """

    def __init__(
        self,
        n: int,
        global_dim: int,
        node_dyn_dim: int,
        node_stat_dim: int,
        hidden_dim: int,
        global_context_dim: int,
        num_nodes: int,
        act_fn: str = "relu",
        output_act_fn: str = "none",
        num_layers: int = 1,
        ignore_global_v_node: bool = True,
        num_message_passing: int = 1,
        aggr_fn: str = "mean",
        layer_norm: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()

        self.ensemble_size = n  # Ensemble size
        self.global_dim = global_dim  # global_dim is for the central agent node
        self.node_dyn_dim = node_dyn_dim
        self.node_stat_dim = node_stat_dim
        self.hidden_dim = hidden_dim
        self.global_context_dim = global_context_dim
        self.num_nodes = num_nodes

        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.num_layers = num_layers
        self.ignore_global_v_node = ignore_global_v_node
        self.num_message_passing = num_message_passing
        self.aggr_fn = aggr_fn
        self.layer_norm = layer_norm

        self.device = device

        # The node attribute dimension per object is dynamic + static components
        node_attr_dim = self.node_dyn_dim + self.node_stat_dim

        mlp_parallel_ensemble_params = {
            "n": self.ensemble_size,
            "num_layers": self.num_layers,
            "size": self.hidden_dim,
            "activation": self.act_fn,
            "output_activation": self.output_act_fn,
            "l1_reg": 0,
            "l2_reg": 0,
            "weight_initializer": "torch_truncated_normal",
            "bias_initializer": "constant_zero",
            "use_spectral_normalization": False,
            "use_layer_normalization": self.layer_norm,
        }

        # Edge MLP takes in the states of neighboring nodes and the agent state and action
        self.edge_mlp = MLPParallelEnsemble(
            input_dim=node_attr_dim * 2 + self.global_dim + self.global_context_dim,
            output_dim=self.hidden_dim,
            params=mlp_parallel_ensemble_params,
        )

        # The node MLP takes as input the node attribute,
        # the edge attributes coming into the node aggregated (dimension: hidden_dim),
        # the action and the global feature
        node_input_dim = node_attr_dim + self.hidden_dim + self.global_context_dim + self.global_dim

        self.node_mlp = MLPParallelEnsemble(
            input_dim=node_input_dim,
            output_dim=self.node_dyn_dim,
            params=mlp_parallel_ensemble_params,
        )

        # Global MLP for the prediction of the agent's state!
        global_input_dim = self.global_dim + self.hidden_dim + self.global_context_dim

        if not self.ignore_global_v_node:
            # the global edge MLP computes object-agent interactions
            self.edge_mlp_global = MLPParallelEnsemble(
                input_dim=node_attr_dim + self.global_dim + self.global_context_dim,
                output_dim=self.hidden_dim,
                params=mlp_parallel_ensemble_params,
            )

            global_input_dim += self.hidden_dim

        # Use the aggregated edge info for the whole graph + agent state and action +
        # (if ignore_global_v_node=False) the agent-object state computations (as if direct edges between these nodes)

        self.global_mlp = MLPParallelEnsemble(
            input_dim=global_input_dim,
            output_dim=self.global_dim,
            params=mlp_parallel_ensemble_params,
        )

        self.edge_list = None
        self.batch_size = 0

        self.to(self.device)

    # Edge function of the GNN
    def _edge_model(self, source, target, context=None):
        out = torch.cat([source, target], dim=-1)
        if context is not None:
            out = torch.cat([out, context], dim=-1)
        return self.edge_mlp(out)

    # Node transition function of the GNN: local mode is for the node updates themselves,
    # global mode is for the global state update, i.e. the agent state in this case
    def _node_model(self, node_attr, edge_index_for_segment, edge_attr, mode="local"):
        # node_attr: [ensemble_size, nB*nodes, node_attr_dim]
        # edge_attr: [ensemble_size, nB*num_edges, hidden_dim)]
        if edge_attr is not None:
            # row, col = edge_index
            if self.aggr_fn == "mean":
                agg = unsorted_segment_mean_ensemble(edge_attr, edge_index_for_segment, num_segments=node_attr.size(1))
            else:
                agg = unsorted_segment_sum_ensemble(edge_attr, edge_index_for_segment, num_segments=node_attr.size(1))
            out = torch.cat([node_attr, agg], dim=-1)
        else:
            out = node_attr
        if mode == "local":
            return self.node_mlp(out)
        elif mode == "global":
            return self.global_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_nodes):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size or self.num_nodes != num_nodes:
            self.batch_size = batch_size
            self.num_nodes = num_nodes
            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_nodes, num_nodes)

            # Remove diagonal.
            adj_full -= torch.eye(num_nodes)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(0, batch_size * num_nodes, num_nodes).unsqueeze(-1)
            offset = offset.expand(batch_size, num_nodes * (num_nodes - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1).to(self.device)

        return self.edge_list

    # Forwards the prior and posterior transition models
    def _forward_transition_gnn(
        self,
        node_attributes: torch.Tensor,
        global_context: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        # cuda = node_attributes.is_cuda
        # node_attributes: [ensemble_size, batch_size, num_nodes, node_dim]
        assert node_attributes.size(0) == self.ensemble_size
        batch_size = node_attributes.size(1)
        num_nodes = node_attributes.size(2)

        # node_attr: Flatten states tensor to [ensemble_size, B * num_nodes, embedding_dim]
        node_attr = node_attributes.reshape(self.ensemble_size, -1, self.node_dyn_dim + self.node_stat_dim)
        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_nodes*[num_nodes-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(batch_size, num_nodes)

            row, col = edge_index
            # Row contains indices of each edge in the graph with an offset between different elements within batch!

            # Find the index of the batch sample for each edge in row:
            # edge_batch_ind = torch.floor_divide(row, num_nodes)
            edge_batch_ind = torch.div(row, num_nodes, rounding_mode="trunc")

            # global_context: [ensemble_size, batch_size, global_dim]
            # action: [ensemble_size, batch_size, action_dim]
            # index_select across batch dimension (axis 1) for each edge -> [ensemble_size, batch_size*num_edges, *]
            global_context_per_edge = torch.index_select(global_context, 1, edge_batch_ind)
            action_per_edge = torch.index_select(action, 1, edge_batch_ind)

            edge_attr = self._edge_model(
                node_attr[:, row, :],
                node_attr[:, col, :],
                context=torch.cat([global_context_per_edge, action_per_edge], dim=-1),
            )

        # Prepare context (agent state and action) for the node transition function!
        # flat_shared: [ensemble_size, nB, ...]
        flat_shared = torch.cat([global_context, action], dim=-1)

        # Node_batch_ind: (N*nObj) x 1
        node_batch_ind = torch.repeat_interleave(torch.arange(0, batch_size, 1), self.num_nodes).to(self.device)

        # batch_shared: (N*nObj) x shared_dim
        batch_shared = torch.index_select(flat_shared, 1, node_batch_ind)

        node_attr = torch.cat([node_attr, batch_shared], dim=-1)

        if not self.ignore_global_v_node:
            # node_attr_global: ensemble_size x (N*nObj) x hidden_dim
            node_attr_global = self.edge_mlp_global(node_attr)

            # aggregate the node_attr per batch sample!
            # Global node aggregation segment ids for (batch_size*nObj, hidden_din)
            if self.aggr_fn == "mean":
                global_node_agg = unsorted_segment_mean_ensemble(
                    node_attr_global, segment_ids=node_batch_ind, num_segments=batch_size
                )
            else:
                global_node_agg = unsorted_segment_sum_ensemble(
                    node_attr_global, segment_ids=node_batch_ind, num_segments=batch_size
                )

        # Local node aggregation segment ids for (batch_size*(nObj-1), hidden_din)
        node_attr = self._node_model(node_attr, row, edge_attr, mode="local")

        # Global state prediction!

        # # Global_edge attribute is the aggregation of all edge attributes of the objects of one sample in a batch:
        # # Size: N x hidden_dim
        # global_edge_agg = unsorted_segment_sum(edge_attr, segment_ids=edge_batch_ind, num_segments=batch_size)
        # global_attr_out = self.global_mlp(torch.cat([global_context, action, global_edge_agg], dim=1))

        global_node_input = torch.cat([global_context, action], dim=-1)
        if not self.ignore_global_v_node:
            global_node_input = torch.cat([global_node_input, global_node_agg], dim=-1)
        # Global edge aggregation segment ids for (batch_size*(no_edges_in_graph), hidden_din)
        global_attr_out = self._node_model(global_node_input, edge_batch_ind, edge_attr, mode="global")

        # [batch_size, num_nodes, hidden_dim]
        node_attr_out = node_attr.view(self.ensemble_size, batch_size, num_nodes, -1)
        return (global_attr_out, node_attr_out)

    def forward(
        self,
        agent_state: torch.Tensor,
        object_dyn_state: torch.Tensor,
        object_stat_state: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next agent state as well as the next object dynamic states"""
        node_attr = object_dyn_state
        global_attr = agent_state

        for _ in range(self.num_message_passing):
            if object_stat_state is not None:
                node_attr_in = torch.cat([node_attr, object_stat_state], dim=-1)
            else:
                node_attr_in = node_attr
            global_attr, node_attr = self._forward_transition_gnn(node_attr_in, global_attr, action)

        return (global_attr, node_attr)


class HomogeneousGraphNeuralNetworkEnsemble(nn.Module):
    """A graph neural network implementation for the dynamics model.
    Code taken and modified from: https://github.com/tkipf/c-swm
    """

    def __init__(
        self,
        n: int,
        agent_dim: int,
        node_dyn_dim: int,
        node_stat_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        action_dim: int,  # for the action
        num_nodes: int,
        act_fn: str = "relu",
        output_act_fn: str = "none",
        num_layers: int = 1,
        num_message_passing: int = 1,
        aggr_fn: str = "mean",
        layer_norm: bool = True,
        embedding: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()

        self.ensemble_size = n  # Ensemble size
        self.agent_dim = agent_dim  # agent_dim is for the central agent node
        self.node_dyn_dim = node_dyn_dim
        self.node_stat_dim = node_stat_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_nodes = num_nodes

        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.num_layers = num_layers
        self.num_message_passing = num_message_passing
        self.aggr_fn = aggr_fn
        self.layer_norm = layer_norm

        if agent_dim > 0 or agent_dim != node_dyn_dim or node_stat_dim > 0:
            self.embedding = True  # In this case embedding is needed!
        elif embedding_dim == node_dyn_dim and agent_dim == 0:
            self.embedding = False
        else:
            self.embedding = embedding

        self.device = device

        # The node attribute dimension per object is dynamic + static components
        obj_node_attr_dim = self.node_dyn_dim + self.node_stat_dim

        mlp_parallel_ensemble_params = {
            "n": self.ensemble_size,
            "num_layers": self.num_layers,
            "size": self.hidden_dim,
            "activation": self.act_fn,
            "output_activation": self.output_act_fn,
            "l1_reg": 0,
            "l2_reg": 0,
            "weight_initializer": "torch_truncated_normal",
            "bias_initializer": "constant_zero",
            "use_spectral_normalization": False,
            "use_layer_normalization": self.layer_norm,
        }

        mlp_parallel_ensemble_params_embedding = mlp_parallel_ensemble_params.copy()
        mlp_parallel_ensemble_params_embedding["num_layers"] = 0
        mlp_parallel_ensemble_params_embedding["activation"] = "none"
        mlp_parallel_ensemble_params_embedding["use_layer_normalization"] = False

        if self.embedding:
            if self.agent_dim > 0:
                self.embed_agent = MLPParallelEnsemble(
                    input_dim=self.agent_dim,
                    output_dim=self.embedding_dim,
                    params=mlp_parallel_ensemble_params_embedding,
                )

            self.embed_obj = MLPParallelEnsemble(
                input_dim=obj_node_attr_dim,
                output_dim=self.embedding_dim,
                params=mlp_parallel_ensemble_params_embedding,
            )

            if self.agent_dim > 0:
                self.output_head_agent = MLPParallelEnsemble(
                    input_dim=self.embedding_dim,
                    output_dim=self.agent_dim,
                    params=mlp_parallel_ensemble_params_embedding,
                )

            self.output_head_objdyn = MLPParallelEnsemble(
                input_dim=self.embedding_dim,
                output_dim=self.node_dyn_dim,
                params=mlp_parallel_ensemble_params_embedding,
            )

        self.feature_dim = self.embedding_dim if self.embedding else self.node_dyn_dim
        # Edge MLP takes in the states of neighboring nodes and the agent state and action
        self.edge_mlp = MLPParallelEnsemble(
            input_dim=self.feature_dim * 2 + self.action_dim,
            output_dim=self.hidden_dim,
            params=mlp_parallel_ensemble_params,
        )

        # The node MLP takes as input the node attribute,
        # the edge attributes coming into the node aggregated (dimension: hidden_dim),
        # the action and the global feature
        node_input_dim = self.feature_dim + self.hidden_dim + self.action_dim

        self.node_mlp = MLPParallelEnsemble(
            input_dim=node_input_dim, output_dim=self.feature_dim, params=mlp_parallel_ensemble_params
        )

        self.edge_list = None
        self.batch_size = 0

        self.to(self.device)

    # Edge function of the GNN
    def _edge_model(self, source, target, context=None):
        out = torch.cat([source, target], dim=-1)
        if context is not None:
            out = torch.cat([out, context], dim=-1)
        return self.edge_mlp(out)

    # Node transition functşion of the GNN: local mode is for the node updates themselves,
    # global mode is for the global state update, i.e. the agent state in this case
    def _node_model(self, node_attr, edge_index_for_segment, edge_attr):
        # node_attr: [ensemble_size, nB*nodes, node_attr_dim]
        # edge_attr: [ensemble_size, nB*num_edges, hidden_dim)]
        if edge_attr is not None:
            # row, col = edge_index
            if self.aggr_fn == "mean":
                agg = unsorted_segment_mean_ensemble(edge_attr, edge_index_for_segment, num_segments=node_attr.size(1))
            else:
                agg = unsorted_segment_sum_ensemble(edge_attr, edge_index_for_segment, num_segments=node_attr.size(1))
            out = torch.cat([node_attr, agg], dim=-1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_nodes):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size or self.num_nodes != num_nodes:
            self.batch_size = batch_size
            self.num_nodes = num_nodes
            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_nodes, num_nodes)

            # Remove diagonal.
            adj_full -= torch.eye(num_nodes)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(0, batch_size * num_nodes, num_nodes).unsqueeze(-1)
            offset = offset.expand(batch_size, num_nodes * (num_nodes - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1).to(self.device)

        return self.edge_list

    # Forwards the prior and posterior transition models
    def _forward_transition_gnn(
        self,
        node_attributes: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        # node_attributes: [ensemble_size, batch_size, num_nodes, node_dim]
        assert node_attributes.size(0) == self.ensemble_size
        batch_size = node_attributes.size(1)
        num_nodes = node_attributes.size(2)

        # node_attr: Flatten states tensor to [ensemble_size, B * num_nodes, embedding_dim or slot_dim]
        node_attr = node_attributes.reshape(self.ensemble_size, -1, self.feature_dim)
        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_nodes*[num_nodes-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(batch_size, num_nodes)

            row, col = edge_index
            # Row contains indices of each edge in the graph with an offset between different elements within batch!

            # Find the index of the batch sample for each edge in row:
            # edge_batch_ind = torch.floor_divide(row, num_nodes)
            edge_batch_ind = torch.div(row, num_nodes, rounding_mode="trunc")

            # action: [ensemble_size, batch_size, action_dim]
            # index_select across batch dimension (axis 1) for each edge -> [ensemble_size, batch_size*num_edges, *]
            action_per_edge = torch.index_select(action, 1, edge_batch_ind)

            edge_attr = self._edge_model(node_attr[:, row, :], node_attr[:, col, :], context=action_per_edge)

        # Node_batch_ind: (N*nObj) x 1
        node_batch_ind = torch.repeat_interleave(torch.arange(0, batch_size, 1), self.num_nodes).to(self.device)

        # batch_shared: (N*nObj) x shared_dim (=action_dim)
        batch_shared = torch.index_select(action, 1, node_batch_ind)

        node_attr = torch.cat([node_attr, batch_shared], dim=-1)

        # Local node aggregation segment ids for (batch_size*(nObj-1), hidden_din)
        node_attr = self._node_model(node_attr, row, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(self.ensemble_size, batch_size, num_nodes, -1)

    def forward(
        self,
        agent_state: Optional[torch.Tensor],
        object_dyn_state: torch.Tensor,
        object_stat_state: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next agent state as well as the next object dynamic states"""
        node_attr = object_dyn_state

        if self.embedding and self.agent_dim > 0:
            agent_node_attr_in = self.embed_agent(agent_state)

        object_state = torch.cat(
            [object_dyn_state, object_stat_state], dim=-1
        )  # if object_stat_state is not None else object_dyn_state
        ensemble_size, batch_size, nObj, _ = object_state.shape
        obj_node_attr_in = self.embed_obj(object_state.view(ensemble_size, batch_size * nObj, -1))

        if self.agent_dim > 0:
            # Concatenate the agent and object attributes to get the full graph nodes!
            node_attr = torch.cat(
                [torch.unsqueeze(agent_node_attr_in, 2), obj_node_attr_in.view(ensemble_size, batch_size, nObj, -1)],
                dim=2,
            )
        else:
            node_attr = obj_node_attr_in.view(ensemble_size, batch_size, nObj, -1)

        for _ in range(self.num_message_passing):
            node_attr = self._forward_transition_gnn(node_attr, action)

        # node_attr: ensemble_size x batch_size x num_nodes x feature_dim
        if self.agent_dim:
            agent_out = node_attr[..., 0, :].clone()
            objects_dyn_out = node_attr[..., 1:, :].clone()
        else:
            agent_out = torch.FloatTensor().to(self.device)
            objects_dyn_out = node_attr

        if self.embedding:
            # Apply output heads!
            if self.agent_dim:
                agent_out = self.output_head_agent(agent_out)
            objects_dyn_out_flat = self.output_head_objdyn(
                objects_dyn_out.reshape(ensemble_size, batch_size * nObj, -1)
            )
            objects_dyn_out = objects_dyn_out_flat.view(ensemble_size, batch_size, nObj, -1)

        return (agent_out, objects_dyn_out)
