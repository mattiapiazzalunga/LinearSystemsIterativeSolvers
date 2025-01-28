import numpy as np
import UpdateStrategy as uS
import scipy as sc
import time as tm


class AlgorithmOutput:
    __relative_error: float
    __number_of_iterations: int
    __time: float
    __method: str

    def __init__(self, exact_solution: np.array, solution: np.array, number_of_iterations: int, start_time: float,
                 end_time: float, method: str):
        self.__set_relative_error(exact_solution, solution)
        self.__set_number_of_iterations(number_of_iterations)
        self.__set_time(start_time, end_time)
        self.__set_method(method)

    def __set_relative_error(self, exact_solution: np.array, solution: np.array):
        self.__relative_error = np.linalg.norm(solution - exact_solution) / np.linalg.norm(exact_solution)

    def get_relative_error(self) -> float:
        return self.__relative_error

    def __set_number_of_iterations(self, number_of_iterations):
        self.__number_of_iterations = number_of_iterations

    def get_number_of_iterations(self) -> int:
        return self.__number_of_iterations

    def __set_method(self, method: str):
        self.__method = method

    def get_method(self) -> str:
        return self.__method

    def __set_time(self, start_time: float, end_time: float):
        self.__time = end_time - start_time

    def get_time(self) -> float:
        return self.__time


class GenericAlgorithm:
    __b_term: np.array
    __x_solution: np.array
    __a_matrix: sc.sparse.csr_matrix
    __tolerance: float

    def __init__(self, a_matrix: sc.sparse.csr_matrix, b_term: np.array, x_solution: np.array, tolerance: float):
        if self.get_matrix_rows(a_matrix) == self.__get_array_length(b_term) and self.__get_array_length(
                b_term) == self.__get_array_length(x_solution):
            if tolerance > 0:
                if GenericAlgorithm.__check_matrix(a_matrix):
                    if GenericAlgorithm.__check_symmetric(a_matrix) and GenericAlgorithm.__check_definite_positive(
                            a_matrix):
                        self.__a_matrix = a_matrix
                        self.__b_term = np.array(b_term)
                        self.__x_solution = x_solution
                        self.__tolerance = tolerance
                    else:
                        raise ValueError("Matrix must be symmetrical and positive defined")
                else:
                    raise TypeError("Input is not a matrix")
            else:
                raise ValueError("Tolerance muSt be grater than 0")
        else:
            raise ValueError("The dimensions don't correspond")

    @staticmethod
    def __check_matrix(matrix: sc.sparse.csr_matrix) -> bool:
        return isinstance(matrix, sc.sparse.csr_matrix)

    @staticmethod
    def __check_symmetric(matrix: sc.sparse.csr_matrix) -> bool:
        matrix_dense = matrix.toarray()
        return np.allclose(matrix_dense, matrix_dense.T)

    @staticmethod
    def __check_definite_positive(matrix: sc.sparse.csr_matrix) -> bool:
        try:
            np.linalg.cholesky(matrix.toarray())
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def get_matrix_rows(matrix: sc.sparse.csr_matrix) -> int:
        return matrix.shape[0]

    @staticmethod
    def __get_array_length(array: np.array) -> int:
        return len(array)

    def __get_solution_length(self) -> int:
        return len(self.__x_solution)

    def __stop_criterion(self, residue: np.array) -> bool:
        return (np.linalg.norm(residue) / np.linalg.norm(self.__b_term)) < self.__tolerance

    def apply_algorithm(self, update_strategy: uS.UpdateStrategy) -> AlgorithmOutput:
        start_time = tm.perf_counter()
        k = 0
        strategy_params = uS.StrategyParameters(self.__a_matrix, self.__b_term, np.zeros(self.__get_solution_length()))
        strategy_params.set_residue(uS.residue(strategy_params.get_a_matrix(), strategy_params.get_x_k(),
                                               strategy_params.get_b_term()))
        if isinstance(update_strategy, uS.ConjugateGradientStrategy):
            strategy_params.set_d_k(strategy_params.get_residue())
        while not (self.__stop_criterion(strategy_params.get_residue())):
            output = update_strategy.update_strategy(strategy_params)
            strategy_params.set_x_k(output.get_x_k_1())
            k = k + 1
            if k > 20000:
                raise RuntimeError("Error, does not converge. Reached 20000 iterations")
            strategy_params.set_residue(output.get_r_k_1())
            if isinstance(update_strategy, uS.ConjugateGradientStrategy):
                strategy_params.set_d_k(output.get_d_k_1())
        end_time = tm.perf_counter()
        return AlgorithmOutput(self.__x_solution, strategy_params.get_x_k(), k, start_time, end_time,
                               type(update_strategy))
