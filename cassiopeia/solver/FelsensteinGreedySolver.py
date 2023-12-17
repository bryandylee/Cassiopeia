from typing import Callable, Dict, List, Optional, Generator, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import random
import math

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import (
    GreedySolverError,
    find_duplicate_groups,
    is_ambiguous_state,
)
from cassiopeia.solver import (
    dissimilarity_functions,
    graph_utilities,
    GreedySolver,
    missing_data_methods,
    solver_utilities,
)


class Node:
    """Initializes a node with given parameters.

    Arguments:
        name: name of node (only relevant for leaves)
        left: left child (Node)
        right: right child (Node)
        branch_length: length of branch that leads to this node (float)
        branch_id: id of branch that leads to this node (int)
        probs: probability of observed bases beneath this node
                [list of 4 probs for 'ACGT'] (initialized to None)
    """

    def __init__(self, name, left, right, branch_length, branch_id=None):
        self.name = name
        self.left = left
        self.right = right
        self.branch_length = branch_length
        self.branch_id = branch_id
        self.probs = [None for _ in range(52)]


def jcm(b, a, t, u=1.0):
    if a == b:
        return (1 / 51) * (1 + 50 * np.exp((-51 / 50) * u * t))
    else:  # when a =/= b
        return (1 / 51) * (1 - np.exp((-51 / 50) * u * t))


def parse_newick(newick):
    def add_node(name, children):
        # If a ':' is found, there is a name and branch length
        if ":" in name:
            name, branch_length = name.split(":")
            branch_length = float(branch_length)
        # If no ':' is found (ex. root), set branch_length to zero
        else:
            branch_length = 0

        left = children[0] if children else None
        right = children[1] if len(children) > 1 else None
        return Node(name, left, right, branch_length)

    stack = []
    current_node = ("", [])
    for char in newick:
        if char == "(":
            stack.append(current_node)
            current_node = ("", [])
        elif char == ",":
            parent = stack[-1]
            node = add_node(*current_node)
            parent[1].append(node)
            current_node = ("", [])
        elif char == ")":
            parent = stack.pop()
            node = add_node(*current_node)
            parent[1].append(node)
            current_node = parent
        elif char == ";":
            break
        else:
            current_node = (current_node[0] + char, current_node[1])

    return add_node(*current_node)


def postorder_traversal(node):
    result = []
    branch_lengths = []

    def traverse(node, result, branch_lengths):
        if node.left:
            traverse(node.left, result, branch_lengths)
        if node.right:
            traverse(node.right, result, branch_lengths)
        node.branch_id = len(result)
        result.append(node)
        branch_lengths.append(node.branch_length)

    traverse(node, result, branch_lengths)
    return result, branch_lengths


def initialize_topology(newick_string):
    tree = parse_newick(newick_string)
    branch_lengths = []
    ordering, branch_lengths = postorder_traversal(tree)
    bases = list(range(-1, 51))
    branch_probs = [
        np.zeros((len(bases), len(bases)), dtype=float)
        for _ in range(len(branch_lengths) + 1)
    ]
    lengths = np.append(branch_lengths, 0)  # to get the zero length branch
    for branch_id, t in enumerate(lengths):
        for ancestor_base, a in enumerate(bases):
            for descendant_base, b in enumerate(bases):
                branch_probs[branch_id][ancestor_base][descendant_base] = jcm(b, a, t)
    return ordering, branch_probs


def sumLogProbs(a, b):
    if a == float("-inf") and b == float("-inf"):
        return float("-inf")
    if a > b:
        return a + np.log1p(math.exp(b - a))
    else:
        return b + np.log1p(math.exp(a - b))


def likelihood(data, seqlen, ordering, bp):
    bases = list(range(-1, 51))
    pi = 1 / 51
    total_log_prob = float("-inf")

    z = 0
    for char in range(seqlen):
        # print(z)
        z += 1
        # Initialization
        for node in ordering:
            if node.left is None and node.right is None:
                for i in bases:  # range(len(bases)):
                    if bases[i] == data[node.name][char]:
                        node.probs[i] = 0.0
                    else:
                        node.probs[i] = float("-inf")

        # Recursion
        for node in ordering:
            if node.left is not None and node.right is not None:
                for parent_ind in bases:  # range(len(bases)):
                    prob_l = float("-inf")
                    prob_r = float("-inf")
                    for child_ind in bases:  # range(len(bases)):
                        # print(prob_l)
                        prob_l = sumLogProbs(
                            prob_l,
                            node.left.probs[child_ind]
                            + np.log(bp[node.left.branch_id][parent_ind][child_ind]),
                        )
                        # print(prob_r)
                        prob_r = sumLogProbs(
                            prob_r,
                            node.right.probs[child_ind]
                            + np.log(bp[node.right.branch_id][parent_ind][child_ind]),
                        )
                        # print(node.left.probs[child_ind]
                        #     + np.log(bp[node.left.branch_id][parent_ind][child_ind]))
                        # print(node.right.probs[child_ind]
                        #     + np.log(bp[node.right.branch_id][parent_ind][child_ind]))
                    node.probs[parent_ind] = prob_l + prob_r
                    # print(node.probs)
        # Termination
        prob_sum = float("-inf")
        for i in bases:  # range(len(bases)):
            prob_sum = sumLogProbs(prob_sum, np.log(pi) + ordering[-1].probs[i])
        total_log_prob = sumLogProbs(total_log_prob, prob_sum)

    return total_log_prob


class FelsensteinGreedySolver(GreedySolver.GreedySolver):

    def __init__(
        self,
        missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
        similarity_function: Optional[
            Callable[
                [
                    List[int],
                    List[int],
                    int,
                    Optional[Dict[int, Dict[int, float]]],
                ],
                float,
            ]
        ] = dissimilarity_functions.hamming_similarity_without_missing,
        threshold: Optional[int] = 0,
        prior_transformation: str = "negative_log",
        hybrid_split_probability: float = 0.05,
    ):
        super().__init__(prior_transformation)

        self.missing_data_classifier = missing_data_classifier
        self.allow_ambiguous = True
        self.hybrid_split_probability = hybrid_split_probability
        self.threshold = threshold
        self.similarity_function = similarity_function

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """Partitions based on the most frequent (character, state) pair.

        Uses the (character, state) pair to split the list of samples into
        two partitions. In doing so, the procedure makes use of the missing
        data classifier to classify samples that have missing data at that
        character where presence or absence of the character is ambiguous.

        Args:
            character_matrix: Character matrix
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partition groups
        """

        sample_indices = solver_utilities.convert_sample_names_to_indices(
            character_matrix.index, samples
        )
        mutation_frequencies = self.compute_mutation_frequencies(
            samples, character_matrix, missing_state_indicator
        )

        if random.random() < self.hybrid_split_probability:
            # Random Splitting
            chosen_character = random.choice(list(mutation_frequencies.keys()))
            chosen_state = random.choice(
                list(mutation_frequencies[chosen_character].keys())
            )
            while chosen_state != missing_state_indicator and chosen_state != 0:
                chosen_state = random.choice(
                    list(mutation_frequencies[chosen_character].keys())
                )
        else:
            best_frequency = 0
            chosen_character = 0
            chosen_state = 0
            for character in mutation_frequencies:
                for state in mutation_frequencies[character]:
                    if state != missing_state_indicator and state != 0:
                        # Avoid splitting on mutations shared by all samples
                        if (
                            mutation_frequencies[character][state]
                            < len(samples)
                            - mutation_frequencies[character][missing_state_indicator]
                        ):
                            if weights:
                                if (
                                    mutation_frequencies[character][state]
                                    * weights[character][state]
                                    > best_frequency
                                ):
                                    chosen_character, chosen_state = (
                                        character,
                                        state,
                                    )
                                    best_frequency = (
                                        mutation_frequencies[character][state]
                                        * weights[character][state]
                                    )
                            else:
                                if (
                                    mutation_frequencies[character][state]
                                    > best_frequency
                                ):
                                    chosen_character, chosen_state = (
                                        character,
                                        state,
                                    )
                                    best_frequency = mutation_frequencies[character][
                                        state
                                    ]

        if chosen_state == 0:
            return samples, []

        left_set = []
        right_set = []
        missing = []

        unique_character_array = character_matrix.to_numpy()
        sample_names = list(character_matrix.index)

        ambiguous_contains = (
            lambda query, _s: _s in query if is_ambiguous_state(query) else _s == query
        )

        for i in sample_indices:
            observed_state = unique_character_array[i, chosen_character]
            if ambiguous_contains(observed_state, chosen_state):
                left_set.append(sample_names[i])
            elif unique_character_array[i, chosen_character] == missing_state_indicator:
                missing.append(sample_names[i])
            else:
                right_set.append(sample_names[i])

        left_set, right_set = self.missing_data_classifier(
            character_matrix,
            missing_state_indicator,
            left_set,
            right_set,
            missing,
            weights=weights,
        )

        return left_set, right_set

    def perform_split_max_cut(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """Performs a partition using both Greedy and MaxCut criteria.

        First, uses the most frequent (character, state) pair to split the list
        of samples. In doing so, the procedure makes use of the missing data
        classifier. Then, it optimizes this partition for the max cut on a
        connectivity graph constructed on the samples using a hill-climbing
        method.

        Args:
            character_matrix: Character matrix
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partition groups
        """

        sample_indices = solver_utilities.convert_sample_names_to_indices(
            character_matrix.index, samples
        )
        mutation_frequencies = self.compute_mutation_frequencies(
            samples, character_matrix, missing_state_indicator
        )

        best_frequency = 0
        chosen_character = 0
        chosen_state = 0
        for character in mutation_frequencies:
            for state in mutation_frequencies[character]:
                if state != missing_state_indicator and state != 0:
                    # Avoid splitting on mutations shared by all samples
                    if (
                        mutation_frequencies[character][state]
                        < len(samples)
                        - mutation_frequencies[character][missing_state_indicator]
                    ):
                        if weights:
                            if (
                                mutation_frequencies[character][state]
                                * weights[character][state]
                                > best_frequency
                            ):
                                chosen_character, chosen_state = (
                                    character,
                                    state,
                                )
                                best_frequency = (
                                    mutation_frequencies[character][state]
                                    * weights[character][state]
                                )
                        else:
                            if mutation_frequencies[character][state] > best_frequency:
                                chosen_character, chosen_state = (
                                    character,
                                    state,
                                )
                                best_frequency = mutation_frequencies[character][state]

        if chosen_state == 0:
            return samples, []

        left_set = []
        right_set = []
        missing = []

        unique_character_array = character_matrix.to_numpy()
        sample_names = list(character_matrix.index)

        for i in sample_indices:
            if unique_character_array[i, chosen_character] == chosen_state:
                left_set.append(sample_names[i])
            elif unique_character_array[i, chosen_character] == missing_state_indicator:
                missing.append(sample_names[i])
            else:
                right_set.append(sample_names[i])

        left_set, right_set = self.missing_data_classifier(
            character_matrix,
            missing_state_indicator,
            left_set,
            right_set,
            missing,
            weights=weights,
        )

        G = graph_utilities.construct_connectivity_graph(
            character_matrix,
            mutation_frequencies,
            missing_state_indicator,
            samples,
            weights=weights,
        )

        improved_left_set = graph_utilities.max_cut_improve_cut(G, left_set)

        improved_right_set = []
        for i in samples:
            if i not in improved_left_set:
                improved_right_set.append(i)

        return improved_left_set, improved_right_set

    def perform_split_spectral(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """Performs a partition using both Greedy and Spectral criteria.

        First, uses the most frequent (character, state) pair to split the
        list of samples. In doing so, the procedure makes use of the missing
        data classifier. Then, it optimizes this partition for the normalized
        cut on a similarity graph constructed on the samples using a hill-
        climbing method.

        Args:
            character_matrix: Character matrix
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partition groups
        """
        sample_indices = solver_utilities.convert_sample_names_to_indices(
            character_matrix.index, samples
        )
        mutation_frequencies = self.compute_mutation_frequencies(
            samples, character_matrix, missing_state_indicator
        )

        best_frequency = 0
        chosen_character = 0
        chosen_state = 0
        for character in mutation_frequencies:
            for state in mutation_frequencies[character]:
                if state != missing_state_indicator and state != 0:
                    # Avoid splitting on mutations shared by all samples
                    if (
                        mutation_frequencies[character][state]
                        < len(samples)
                        - mutation_frequencies[character][missing_state_indicator]
                    ):
                        if weights:
                            if (
                                mutation_frequencies[character][state]
                                * weights[character][state]
                                > best_frequency
                            ):
                                chosen_character, chosen_state = (
                                    character,
                                    state,
                                )
                                best_frequency = (
                                    mutation_frequencies[character][state]
                                    * weights[character][state]
                                )
                        else:
                            if mutation_frequencies[character][state] > best_frequency:
                                chosen_character, chosen_state = (
                                    character,
                                    state,
                                )
                                best_frequency = mutation_frequencies[character][state]

        if chosen_state == 0:
            return samples, []

        left_set = []
        right_set = []
        missing = []

        unique_character_array = character_matrix.to_numpy()
        sample_names = list(character_matrix.index)

        for i in sample_indices:
            if unique_character_array[i, chosen_character] == chosen_state:
                left_set.append(sample_names[i])
            elif unique_character_array[i, chosen_character] == missing_state_indicator:
                missing.append(sample_names[i])
            else:
                right_set.append(sample_names[i])

        left_set, right_set = self.missing_data_classifier(
            character_matrix,
            missing_state_indicator,
            left_set,
            right_set,
            missing,
            weights=weights,
        )

        G = graph_utilities.construct_similarity_graph(
            character_matrix,
            missing_state_indicator,
            samples,
            similarity_function=self.similarity_function,
            threshold=self.threshold,
            weights=weights,
        )

        improved_left_set = graph_utilities.spectral_improve_cut(G, left_set)

        improved_right_set = []
        for i in samples:
            if i not in improved_left_set:
                improved_right_set.append(i)

        return improved_left_set, improved_right_set

    def solve_helper(
        self,
        split_func,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ):
        """Implements a top-down greedy solving procedure.

        The procedure recursively splits a set of samples to build a tree. At
        each partition of the samples, an ancestral node is created and each
        side of the partition is placed as a daughter clade of that node. This
        continues until each side of the partition is comprised only of single
        samples. If an algorithm cannot produce a split on a set of samples,
        then those samples are placed as sister nodes and the procedure
        terminates, generating a polytomy in the tree. This function will
        populate a tree inside the input CassiopeiaTree.

        Args:
            cassiopeia_tree: CassiopeiaTree storing a character matrix and
                priors.
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
            collapse_mutationless_edges: Indicates if the final reconstructed
                tree should collapse mutationless edges based on internal states
                inferred by Camin-Sokal parsimony. In scoring accuracy, this
                removes artifacts caused by arbitrarily resolving polytomies.
            logfile: File location to log output. Not currently used.
        """

        # A helper function that builds the subtree given a set of samples
        def _solve(
            samples: List[Union[str, int]],
            tree: nx.DiGraph,
            unique_character_matrix: pd.DataFrame,
            weights: Dict[int, Dict[int, float]],
            missing_state_indicator: int,
        ):
            if len(samples) == 1:
                return samples[0]
            # Finds the best partition of the set given the split criteria
            clades = list(
                split_func(
                    unique_character_matrix,
                    samples,
                    weights,
                    missing_state_indicator,
                )
            )
            # Generates a root for this subtree with a unique int identifier
            root = next(node_name_generator)
            tree.add_node(root)

            for clade in clades:
                if len(clade) == 0:
                    clades.remove(clade)

            # If unable to return a split, generate a polytomy and return
            if len(clades) == 1:
                for clade in clades[0]:
                    tree.add_edge(root, clade)
                return root
            # Recursively generate the subtrees for each daughter clade
            for clade in clades:
                child = _solve(
                    clade,
                    tree,
                    unique_character_matrix,
                    weights,
                    missing_state_indicator,
                )
                tree.add_edge(root, child)
            return root

        node_name_generator = solver_utilities.node_name_generator()

        weights = None
        if cassiopeia_tree.priors:
            weights = solver_utilities.transform_priors(
                cassiopeia_tree.priors, self.prior_transformation
            )

        # extract character matrix
        if layer:
            character_matrix = cassiopeia_tree.layers[layer].copy()
        else:
            character_matrix = cassiopeia_tree.character_matrix.copy()

        # Raise exception if the character matrix has ambiguous states.
        if (
            any(
                is_ambiguous_state(state) for state in character_matrix.values.flatten()
            )
            and not self.allow_ambiguous
        ):
            raise GreedySolverError(
                "Ambiguous states are not currently supported with this solver."
            )

        keep_rows = (
            character_matrix.apply(
                lambda x: [
                    set(s) if is_ambiguous_state(s) else set([s]) for s in x.values
                ],
                axis=0,
            )
            .apply(tuple, axis=1)
            .drop_duplicates()
            .index.values
        )
        unique_character_matrix = character_matrix.loc[keep_rows].copy()

        tree = nx.DiGraph()
        tree.add_nodes_from(list(unique_character_matrix.index))

        _solve(
            list(unique_character_matrix.index),
            tree,
            unique_character_matrix,
            weights,
            cassiopeia_tree.missing_state_indicator,
        )

        # Append duplicate samples
        duplicates_tree = self.__add_duplicates_to_tree(
            tree, character_matrix, node_name_generator
        )
        cassiopeia_tree.populate_tree(duplicates_tree, layer=layer)

        # Collapse mutationless edges
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(infer_ancestral_characters=True)

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ):
        vanilla_tree = cassiopeia_tree.copy()
        max_cut_tree = cassiopeia_tree.copy()
        spectral_tree = cassiopeia_tree.copy()

        self.solve_helper(
            self.perform_split, vanilla_tree, layer, collapse_mutationless_edges
        )
        self.solve_helper(
            self.perform_split_max_cut, max_cut_tree, layer, collapse_mutationless_edges
        )
        self.solve_helper(
            self.perform_split_spectral,
            spectral_tree,
            layer,
            collapse_mutationless_edges,
        )

        highest_likelihood = float("-inf")
        highest_ind = -1
        pos_trees = [vanilla_tree, max_cut_tree, spectral_tree]

        for ind, tree in enumerate(pos_trees):
            data = {
                k: v.tolist()
                for v, k in zip(
                    tree.character_matrix.values,
                    tree.character_matrix.index,
                )
            }
            seqlen = len(tree.character_matrix.values[0])
            orderings, probs = initialize_topology(
                tree.get_newick(record_branch_lengths=True)
            )
            pos_likelihood = likelihood(data, seqlen, orderings, probs)
            if highest_likelihood <= pos_likelihood:
                highest_likelihood = pos_likelihood
                highest_ind = ind

        if highest_ind == 0:
            self.solve_helper(
                self.perform_split, cassiopeia_tree, layer, collapse_mutationless_edges
            )
        elif highest_ind == 1:
            self.solve_helper(
                self.perform_split_max_cut,
                cassiopeia_tree,
                layer,
                collapse_mutationless_edges,
            )
        else:
            self.solve_helper(
                self.perform_split_spectral,
                cassiopeia_tree,
                layer,
                collapse_mutationless_edges,
            )

    def __add_duplicates_to_tree(
        self,
        tree: nx.DiGraph,
        character_matrix: pd.DataFrame,
        node_name_generator: Generator[str, None, None],
    ) -> nx.DiGraph:
        """Takes duplicate samples and places them in the tree.

        Places samples removed in removing duplicates in the tree as sisters
        to the corresponding cells that share the same mutations.

        Args:
            tree: The tree to have duplicates added to
            character_matrix: Character matrix

        Returns:
            The tree with duplicates added
        """

        duplicate_mappings = find_duplicate_groups(character_matrix)

        for i in duplicate_mappings:
            new_internal_node = next(node_name_generator)
            nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
            for duplicate in duplicate_mappings[i]:
                tree.add_edge(new_internal_node, duplicate)

        return tree
