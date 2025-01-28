from abc import ABC, abstractmethod
import numpy as np
import scipy as sc


class StrategyParameters:
    __p_reverse_matrix: sc.sparse.csr_matrix
    __a_matrix: sc.sparse.csr_matrix
    __residue: np.array
    __lower_triangular_matrix: sc.sparse.csr_matrix
    __d_k: np.array
    __b_term: np.array
    __x_k: np.array

    def __init__(self, a_matrix: sc.sparse.csr_matrix, b_term: np.array, x_k: np.array):
        self.__set_a_matrix(a_matrix)
        self.__set_p_reverse_matrix()
        self.__set_lower_triangular_matrix()
        self.__set_b_term(b_term)
        self.set_x_k(x_k)

    def __set_b_term(self, b_term: np.array):
        self.__b_term = b_term

    def get_b_term(self) -> np.array:
        return self.__b_term

    def set_x_k(self, x_k: np.array):
        self.__x_k = x_k

    def get_x_k(self) -> np.array:
        return self.__x_k

    def set_d_k(self, d_k: np.array):
        self.__d_k = d_k

    def get_d_k(self) -> np.array:
        return self.__d_k

    def __set_p_reverse_matrix(self):
        p_matrix = sc.sparse.diags(1.0 / self.__a_matrix.diagonal()).tocsr()
        self.__p_reverse_matrix = p_matrix

    def get_p_reverse_matrix(self) -> sc.sparse.csr_matrix:
        return self.__p_reverse_matrix

    def set_residue(self, residue_: np.array):
        self.__residue = residue_

    def get_residue(self) -> np.array:
        return self.__residue

    def __set_lower_triangular_matrix(self):
        self.__lower_triangular_matrix = sc.sparse.tril(self.__a_matrix, k=0).tocsr()

    def get_lower_triangular_matrix(self) -> sc.sparse.csr_matrix:
        return self.__lower_triangular_matrix

    def get_a_matrix(self) -> sc.sparse.csr_matrix:
        return self.__a_matrix

    def __set_a_matrix(self, a_matrix: sc.sparse.csr_matrix):
        self.__a_matrix = a_matrix


class StrategyOutput:
    __x_k_1: np.array
    __r_k_1: np.array
    __d_k_1: np.array

    def set_x_k_1(self, x_k_1):
        self.__x_k_1 = x_k_1

    def get_x_k_1(self) -> np.array:
        return self.__x_k_1

    def set_r_k_1(self, r_k_1):
        self.__r_k_1 = r_k_1

    def get_r_k_1(self) -> np.array:
        return self.__r_k_1

    def set_d_k_1(self, d_k_1):
        self.__d_k_1 = d_k_1

    def get_d_k_1(self) -> np.array:
        return self.__d_k_1


class UpdateStrategy(ABC):
    @abstractmethod
    def update_strategy(self, params: StrategyParameters) -> StrategyOutput:
        pass


class JacobyStrategy(UpdateStrategy):
    output = StrategyOutput()

    def update_strategy(self, params: StrategyParameters) -> StrategyOutput:
        self.output.set_x_k_1(np.add(params.get_x_k(), (params.get_p_reverse_matrix().dot(params.get_residue()))))
        self.output.set_r_k_1(residue(params.get_a_matrix(), self.output.get_x_k_1(), params.get_b_term()))
        return self.output


class GaubSeidelStrategy(UpdateStrategy):
    output = StrategyOutput()

    def update_strategy(self, params: StrategyParameters) -> StrategyOutput:
        y = sc.sparse.linalg.spsolve_triangular(params.get_lower_triangular_matrix(), params.get_residue())
        self.output.set_x_k_1(params.get_x_k() + y)
        self.output.set_r_k_1(residue(params.get_a_matrix(), self.output.get_x_k_1(), params.get_b_term()))
        return self.output


class GradientStrategy(UpdateStrategy):
    output = StrategyOutput()

    def update_strategy(self, params: StrategyParameters) -> StrategyOutput:
        r_k_t = np.transpose(params.get_residue())
        y_k = params.get_a_matrix().dot(params.get_residue())
        a = r_k_t.dot(params.get_residue())
        b = r_k_t.dot(y_k)
        a_k = np.divide(a, b)
        self.output.set_x_k_1(np.add(params.get_x_k(), (a_k * params.get_residue())))
        self.output.set_r_k_1(residue(params.get_a_matrix(), self.output.get_x_k_1(), params.get_b_term()))
        return self.output


class ConjugateGradientStrategy(UpdateStrategy):
    output = StrategyOutput()

    def update_strategy(self, params: StrategyParameters) -> StrategyOutput:
        y_k = params.get_a_matrix().dot(params.get_d_k())
        alpha_k = np.divide(params.get_d_k().dot(params.get_residue()), params.get_d_k().dot(y_k))

        self.output.set_x_k_1(np.add(params.get_x_k(), (alpha_k * params.get_d_k())))
        self.output.set_r_k_1(residue(params.get_a_matrix(), self.output.get_x_k_1(), params.get_b_term()))
        w_k = params.get_a_matrix().dot(self.output.get_r_k_1())
        beta_k = np.divide(params.get_d_k().dot(w_k), params.get_d_k().dot(y_k))
        self.output.set_d_k_1(self.output.get_r_k_1() - (beta_k * params.get_d_k()))
        return self.output


def residue(a_matrix: sc.sparse.csr_matrix, x_k: np.array, b_term: np.array) -> np.array:
    temp_multiplication = a_matrix.dot(x_k)
    return b_term - temp_multiplication
