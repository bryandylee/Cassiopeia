from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import math

from cassiopeia.mixins.utilities import is_ambiguous_state
from cassiopeia.solver import (
    GreedySolver,
    missing_data_methods,
    solver_utilities,
)


class EntropyGreedySolver(GreedySolver.GreedySolver):
    def __init__(
        self,
        missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
        prior_transformation: str = "negative_log",
    ):
        super().__init__(prior_transformation)

        self.missing_data_classifier = missing_data_classifier
        self.allow_ambiguous = True

    def compute_entropy(self, frequencies, total_samples):
        """Compute the entropy of a split."""
        entropy = 0
        for freq in frequencies.values():
            if freq > 0:
                probability = freq / total_samples
                entropy -= probability * math.log(probability, 2)
        return entropy

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """Partitions based on entropy.

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

        best_entropy = float("inf")
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
                        current_entropy = self.compute_entropy(
                            mutation_frequencies[character], len(samples)
                        )
                        if current_entropy < best_entropy:
                            chosen_character, chosen_state = character, state
                            best_entropy = current_entropy
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
