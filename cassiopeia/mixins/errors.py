class CassiopeiaTreeError(Exception):
    """An Exception class for the CassiopeiaTree class."""

    pass


class DataSimulatorError(Exception):
    """Generic error for the DataSimulator subclasses"""

    pass


class DistanceSolverError(Exception):
    """An Exception class for all DistanceSolver subclasses."""

    pass


class GreedySolverError(Exception):
    pass


class HybridSolverError(Exception):
    """An Exception class for all HybridSolver subclasses."""

    pass


class ILPSolverError(Exception):
    """An Exception class for all ILPError subclasses."""

    pass


class iTOLError(Exception):
    pass


class LeafSubsamplerError(Exception):
    """An Exception class for the LeafSubsampler class."""

    pass


class PreprocessError(Exception):
    pass


class PriorTransformationError(Exception):
    """An Exception class for generating weights from priors."""

    pass


class SharedMutationJoiningSolverError(Exception):
    """An Exception class for SharedMutationJoiningSolver."""

    pass


class TreeSimulatorError(Exception):
    """An Exception class for all exceptions generated by
    TreeSimulator or a subclass of TreeSimulator
    """

    pass


class UnknownCigarStringError(Exception):
    pass


class UnspecifiedConfigParameterError(Exception):
    pass