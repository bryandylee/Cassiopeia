from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from cassiopeia.mixins.utilities import is_ambiguous_state
from cassiopeia.solver import GreedySolver, missing_data_methods, solver_utilities

class MultiThreadGreedySolver(GreedySolver.GreedySolver):
    def __init__(
        self,
        missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
        prior_transformation: str = "negative_log",
    ):
        super().__init__(prior_transformation)
        self.missing_data_classifier = missing_data_classifier
        self.allow_ambiguous = True

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
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
                            - mutation_frequencies[character][
                                missing_state_indicator
                            ]
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
                                    best_frequency = mutation_frequencies[
                                        character
                                    ][state]

            if chosen_state == 0:
                return samples, []

            left_set = []
            right_set = []
            missing = []

            unique_character_array = character_matrix.to_numpy()
            sample_names = list(character_matrix.index)

            ambiguous_contains = lambda query, _s: _s in query if is_ambiguous_state(query) else _s == query

            for i in sample_indices:
                observed_state = unique_character_array[i, chosen_character]
                if ambiguous_contains(observed_state, chosen_state):
                    left_set.append(sample_names[i])
                elif (
                    unique_character_array[i, chosen_character]
                    == missing_state_indicator
                ):
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

    def compute_mutation_frequencies(
        self, samples, character_matrix, missing_state_indicator
    ):
        frequencies = {}

        def compute_character_frequency(char):
            state_counts = character_matrix.loc[samples, char].value_counts()
            state_counts[missing_state_indicator] = state_counts.get(missing_state_indicator, 0)
            return char, state_counts.to_dict()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compute_character_frequency, char) for char in character_matrix.columns]
            for future in futures:
                char, freq = future.result()
                frequencies[char] = freq

        return frequencies
