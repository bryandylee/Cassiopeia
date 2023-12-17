import random
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from cassiopeia.mixins.utilities import is_ambiguous_state
from cassiopeia.solver import (
    GreedySolver,
    missing_data_methods,
    solver_utilities,
)


class RandomHybridGreedySolver(GreedySolver.GreedySolver):
    def __init__(
        self,
        missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
        prior_transformation: str = "negative_log",
        hybrid_split_probability: float = 0.2,
    ):
        super().__init__(prior_transformation)

        self.missing_data_classifier = missing_data_classifier
        self.allow_ambiguous = True
        self.hybrid_split_probability = hybrid_split_probability

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """Partitions based on most frequent (character, state) pair, randomized
        occasionally.

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

        best_frequency = 0
        chosen_character = 0
        chosen_state = 0

        if random.random() < self.hybrid_split_probability:
            # Random Splitting with conditions
            valid_characters = [
                char
                for char in mutation_frequencies
                if any(
                    state != missing_state_indicator
                    and state != 0
                    and mutation_frequencies[char][state]
                    < len(samples) - mutation_frequencies[char][missing_state_indicator]
                    for state in mutation_frequencies[char]
                )
            ]
            if not valid_characters:
                # Handle the case where no valid characters are found
                return samples, []

            chosen_character = random.choice(valid_characters)
            valid_states = [
                state
                for state in mutation_frequencies[chosen_character]
                if state != missing_state_indicator
                and state != 0
                and mutation_frequencies[chosen_character][state]
                < len(samples)
                - mutation_frequencies[chosen_character][missing_state_indicator]
            ]
            chosen_state = random.choice(valid_states)

        else:
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
