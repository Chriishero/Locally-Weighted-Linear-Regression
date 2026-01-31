import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Self


class LocallyWeightedLinearRegression:
	def __init__(self, n_iteration: int = 100, learning_rate: float = 0.1,
			  tau: float = 0.5, method: str = "normal_equation",
			  verbosity: int = 1) -> None:
		self.__n_iteration = n_iteration
		self.__learning_rate = learning_rate
		self.__tau = tau
		self.__method = method
		self.__verbosity = verbosity

		self.__m: float = 0
		self.__n: float = 0
		self.__cost_history = np.zeros((0, self.__n_iteration))

	m = property(lambda self: self.__m)
	n = property(lambda self: self.__n)
	cost_history = property(lambda self: self.__cost_history)
	
	def predict(self, X: ArrayLike, y: ArrayLike, x: ArrayLike) -> ArrayLike:
		X = np.array(X)
		y = np.array(y).reshape(-1, 1)
		x = x.reshape(1, -1)

		if not self._has_intercept(X):
			X = np.hstack([np.ones((X.shape[0], 1)), X])
		if x.shape[1] != X.shape[1]:
			x = np.hstack([np.ones((1, 1)), x]).reshape(1, -1)

		W = self._compute_weight(X, x)
		theta = self._compute_theta(X, y, W)

		return self._hypothesis(x, theta)
	
	def get_params(self) -> Dict:
		return {
			"n_iteration": self.__n_iteration,
			"learning_rate": self.__learning_rate,
			"tau": self.__tau,
			"learning_method": self.__method,
			"verbosity": self.__verbosity
		}
	
	def _has_intercept(self, X: ArrayLike) -> bool:
		return np.allclose(X[:, 0], 1)
	
	def _hypothesis(self, X: ArrayLike, theta: ArrayLike) -> ArrayLike:
		return X @ theta
	
	def _compute_weight(self, X: ArrayLike, x: ArrayLike) -> ArrayLike:
		weight = np.exp(-np.sum((X - x) ** 2, axis=1) / (2 * self.__tau ** 2))
		return np.diag(weight)
	
	def _loss_function(self, hypothesis: ArrayLike, y: ArrayLike,
					weight: ArrayLike) -> float:
		err = hypothesis - y
		return 1/(2 * self.m) * np.sum(err.T @ weight @ err)
	
	def _compute_gradient(self, X: ArrayLike, y: ArrayLike,
					   hypothesis: ArrayLike, weight: ArrayLike) -> ArrayLike:
		return 1/self.m * (X.T @ weight @ (hypothesis - y))
	
	def _normal_equation(self, X: ArrayLike, y: ArrayLike,
					  weight: ArrayLike) -> ArrayLike:
		return np.linalg.pinv(X.T @ weight @ X) @ X.T @ weight @ y

	def _compute_theta(self, X: ArrayLike, y: ArrayLike, weight: ArrayLike) -> None:
		self.__m = X.shape[0]
		self.__n = X.shape[1]
		theta = np.zeros((self.n, 1))

		if self.__method == "gradient_descent":
			if self.__verbosity > 0:
				print("LocallyWeightedLinearRegression: Gradient Descent")

			try:
				for i in range(0, self.__n_iteration):
					hypothesis = self._hypothesis(X, theta)
					gradient = self._compute_gradient(X, y, hypothesis, weight)
					theta -= self.__learning_rate * gradient
					self._log_cost(self._hypothesis(X, theta), y, weight, iter=i)
			except Exception as e:
				print(f"An error occured: {e}")

		elif self.__method == "normal_equation":
			if self.__verbosity > 0:
				print("LocallyWeightedLinearRegression: Normal Equation")

			try:
				theta = self._normal_equation(X, y, weight)
				self._log_cost(self._hypothesis(X, theta), y, weight, iter=None)

			except Exception as e:
				print(f"An error occured: {e}")

		else:
			raise ValueError(f"Unexpected 'self.__method' value: {self.__method}")
		
		return theta
	
	def _log_cost(self, hypothesis: ArrayLike, y: ArrayLike,
			   weight: ArrayLike, iter: int | None) -> None:
		if iter is None or iter == 0:
			new_row = np.zeros((1, self.__n_iteration))
			self.__cost_history = np.vstack([self.__cost_history, new_row])
		if iter is None:
			self.__cost_history[-1, 0] = self._loss_function(hypothesis, y, weight)
		else:
			self.__cost_history[-1, iter] = self._loss_function(hypothesis, y, weight)

		if self.__verbosity == 2:
			if iter is None:
				print(f"Normal Equation, Loss : {self.cost_history[-1, 0]}")
			else:
				print(f"Gradient Descent, iter {iter} - Loss : {self.cost_history[-1, iter]}")

	@classmethod
	def create_predictor(cls, n_iteration: int = 100, learning_rate: float = 0.1,
			  tau: float = 1, method: str = "gradient_descent",
			  verbosity: int = 1) -> Self:
		if n_iteration < 0:
			raise ValueError("Hyperparameter 'n_iteration' cannot be inferior to 0")

		if learning_rate < 0:
			raise ValueError("Hyperparameter 'learning_rate' cannot be inferior to 0")
		
		if tau < 0:
			raise ValueError("tau 'learning_rate' cannot be inferior to 0")

		if method not in ["gradient_descent", "normal_equation"]:
			if not isinstance(method, str):
				raise TypeError("Hyperparameter 'method' must be a string")
			raise ValueError("Hyperparameter 'method' cannot be different of 'gradient_descent' or 'normal_equation'")

		if verbosity < 0 or verbosity > 2:
			raise ValueError("Hyperparameter 'verbosity' must be between 0 and 2")
		
		return cls(n_iteration, learning_rate, tau, method, verbosity)