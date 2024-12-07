import numpy as np


class Polynomial:
    """
    Clase que representa un polinomio

    Attributes:
        coefficients (list[float]): Lista de coeficientes del polinomio

    Methods:
        __init__(self, coefficients): Inicializa un polinomio con una lista de coeficientes
        __str__(self): Devuelve una representación en cadena del polinomio
        __add__(self, poly): Suma dos polinomios
        __mul__(self, poly): Multiplica dos polinomios
    """

    def __init__(self, coefficients=None):
        self.coefficients = coefficients or [0]

    def __str__(self):
        """
        Devuelve una representación en cadena del polinomio

        Returns:
            str: Representación del polinomio en forma de string
        """
        grade = len(self.coefficients)  # grade - 1 es el grado del polinomio

        # Se crean los términos del polinomio en formato 'coeficiente x^exponente'
        terms = [str(self.coefficients[0])] + [f"{self.coefficients[i]}λ^{i}" for i in range(1, grade)]
        # Se juntan los terminos con la suma
        return " + ".join(terms)

    def __add__(self, poly):
        """
        Suma dos polinomios

        Args:
            poly (Polynomial): Polinomio a sumar

        Returns:
            Polynomial: Polinomio resultante de la suma de la instancia actual y 'poly'
        """
        n = len(poly.coefficients)  # Número de coeficientes del segundo polinomio
        m = len(self.coefficients)  # Número de coeficientes del primer polinomio

        grade = max(len(self.coefficients), len(poly.coefficients))  # grade - 1 es el grado del polinomio
        new_coefficients = [0] * grade

        # Suma los coeficientes de los términos comunes de ambos polinomios
        for i in range(min(n, m)):
            new_coefficients[i] = poly.coefficients[i] + self.coefficients[i]

        # Si el segundo polinomio tiene más términos que el primero, agrega los restantes
        for i in range(min(n, m), len(poly.coefficients)):
            new_coefficients[i] = poly.coefficients[i]

        # Si el primer polinomio tiene más términos que el segundo, agrega los restantes
        for i in range(min(n, m), len(self.coefficients)):
            new_coefficients[i] = self.coefficients[i]

        return Polynomial(new_coefficients)  # Devuelva un nuevo objeto Polynomial con los coeficientes sumados

    def __mul__(self, poly):
        """
        Multiplica dos polinomios

        Args:
            poly (Polynomial): Polinomio a multiplicar

        Returns:
            Polynomial: Polinomio resultante de la multiplicación de la instancia actual y 'poly'
        """
        m = len(self.coefficients)  # Número de coeficientes del primer polinomio
        n = len(poly.coefficients)  # Número de coeficientes del segundo polinomio

        grade = n + m - 1  # grade - 1 es el grado del polinomio
        new_coefficients = [0] * grade

        # Realiza la multiplicación de polinomios usando el método distributivo
        for i in range(n):
            for j in range(m):
                new_coefficients[i + j] += poly.coefficients[i] * self.coefficients[j]

        return Polynomial(
            new_coefficients)  # Devuelve un nuevo objeto Polynomial con los coeficientes de la multiplicacion


def gauss_elimination(A: list[list[float]]):
    """
    Función que implementa el método de eliminación gaussiana para resolver sistemas de ecuaciones lineales
    A es una lista de listas que representa la matriz aumentada del sistema (coeficientes + términos independientes)
    Si el sistema es homogeneo y tiene solucion no trivial, devuelve solucion no trivial

    Args:
        A (list[list[float]]): Matriz aumentada del sistema de ecuaciones

    Returns:
        list[float]: Vector solución del sistema.
    """
    # Definimos una pequeña tolerancia para considerar un número cercano a cero como cero.
    eps = 1e-9

    # Número de variables (o ecuaciones) en el sistema
    n = len(A)
    # Lista que indica en qué fila se encuentra el pivote de cada columna (inicialmente ninguna)
    where = [-1] * n

    # Inicializamos la fila en la que trabajaremos
    row = 0

    # Iteramos por cada columna de la matriz (correspondiente a cada variable)
    for col in range(n):
        selected = row

        # Encontramos la fila con el mayor valor absoluto en la columna actual (para mayor estabilidad numérica)
        for i in range(row, n):
            if abs(A[i][col]) > abs(A[selected][col]):
                selected = i

        # Si el mayor valor absoluto en la columna es menor que la tolerancia, consideramos la columna como cero
        # y si es el caso de que es 0, no hay pivote para la variable de la actual columna, por lo que pase a la siguiente columna
        if abs(A[selected][col]) < eps:
            continue

        # Si la fila seleccionada no es la fila actual, intercambiamos las filas
        if selected != row:
            A[selected], A[row] = A[row], A[selected]

        where[col] = row  # Indicamos que esta columna tiene su pivote en la fila 'row'

        # Eliminamos los valores debajo y encima del pivote para convertir la matriz en forma escalonada reducida.
        for i in range(n):
            if i != row:  # Ignoramos la fila actual

                # Calculamos el factor para dejar en 0 la columna de la variable, ignorando la fila actual
                factor = A[i][col] / A[row][col]

                for j in range(col, n + 1):  # incluye la columna de términos independientes.
                    A[i][j] -= factor * A[row][j]

        row += 1

    # Inicializamos el vector solución con ceros
    sol = [0] * n
    # Identificamos las variables libres (aquellas que no tienen una fila asociada)
    free_vars = [i for i in range(n) if where[i] == -1]

    # Asignamos un valor arbitrario (1 en este caso) a las variables libres para tener solucion no trivial
    for i in free_vars:
        sol[i] = 1

    # Sustitucion hacia atras para hallar cada termino de la solucion
    for i in range(n):
        if where[i] != -1:
            row = where[i]
            sol[i] = -sum(A[row][k] * sol[k] for k in range(i + 1, n)) / A[row][i]

    # Retornamos el vector solución del sistema.
    return sol


def determinant(A: list[list[Polynomial]]):
    """
    Función para calcular el determinante de una matriz, donde los elementos son objetos de la clase Polynomial.
    A es una lista de listas que representa la matriz cuadrada.

    Args:
        A (list[list[Polynomial]]): Matriz cuadrada de entrada

    Returns:
        Polynomial: Determinante de la matriz A.
    """

    n = len(A)  # Dimensión de la matriz

    # Caso base: si la matriz es de 1x1, el determinante es el único elemento de la matriz
    if n == 1:
        return A[0][0]

    # Calculamos la suma de los cofactores
    acum = Polynomial([0])  # inicializar la suma a 0
    for i in range(n):
        sign = Polynomial([(-1) ** i])  # Signo cofactor

        # Construimos la submatriz excluyendo la fila 0 (primera fila) y la columna 'i'
        new_matrix = [[A[x][y] for y in range(n) if y != i] for x in range(n) if x != 0]

        # Calculamos el cofactor multiplicando el signo, el elemento correspondiente, y el determinante de la submatriz
        cofactor = sign * A[0][i] * determinant(new_matrix)

        # Sumamos el cofactor al acumulador del determinante.
        acum = acum + cofactor

    # Devolvemos el determinante calculado
    return acum


def characteristic_equation(A: list[list[float]]):
    """
    Función para calcular el polinomio característico de una matriz cuadrada A
    A es una lista de listas que representa la matriz de entrada (sus elementos son números reales

    Args:
        A (list[list[float]]): Matriz cuadrada de entrada

    Returns:
        Polynomial: Polinomio característico de la matriz A
    """

    n = len(A)  # Dimensión de la matriz

    # Convertimos la matriz A en una matriz de polinomios, donde cada elemento es un objeto Polynomial
    A_poly = [[Polynomial([coefficient]) for coefficient in row] for row in A]

    # Creamos el término identity_lambda (polinomio de la forma 0 - lambda, equivalente a [0, -1])
    identity_lambda = Polynomial([0, -1])

    # Añadimos el término 0 - lambda a cada elemento de la diagonal principal de la matriz A_poly
    for i in range(n):
        A_poly[i][i] = A_poly[i][i] + identity_lambda

    # Calculamos el determinante de la matriz resultante (A - lambda I) para obtener el polinomio característico
    return determinant(A_poly)


def eigenvalues_characteristic_polynomial_method(A: list[list[float]], show=False):
    """
    Función para calcular los valores propios (eigenvalues) de una matriz cuadrada A
    A es una lista de listas que representa la matriz de entrada (sus elementos son números reales)

    Args:
        A (list[list[float]]): Matriz cuadrada de entrada

    Returns:
        list[float]: Lista de valores propios (eigenvalues) de la matriz A
    """

    n = len(A)  # Dimensión de la matriz

    # Calculamos el polinomio característico de la matriz A.
    char_eq: Polynomial = characteristic_equation(A)
    if show:
        print("El polinomio caracteristico de la matriz es ", char_eq)

    # Usamos np.roots para encontrar las raíces del polinomio característico
    roots = np.roots(char_eq.coefficients[::-1])  # los coeficientes en Polynomial estan de menor a mayor

    # Filtramos las raíces para quedarnos solo con las que son reales.
    eigenvalues = sorted([r for r in roots if np.isclose(r.imag, 0)])

    if show:
        print("Los valores propios son ", *[round(x, 2) for x in eigenvalues])

    if show:
        print()

    # Devolvemos los valores propios ordenados (solo los valores propios que son reales).
    return eigenvalues


def eigenvector(A: list[list[float]], eigenvalue: float, show=False):
    """
    Función para calcular un vector propio (eigenvector) asociado a un valor propio (eigenvalue) de una matriz cuadrada A.
    A es una lista de listas que representa la matriz de entrada (sus elementos son números reales o flotantes)
    eigenvalue es el valor propio para el cual se calcula el vector propio

    Args:
        A (list[list[float]]): Matriz cuadrada de entrada
        eigenvalue (float): Valor propio para el cual se calcula el vector propio

    Returns:
        list[float]: Vector propio normalizado asociado al valor propio dado.
    """

    n = len(A)  # Dimensión de la matriz

    # Creamos una matriz aumentada inicializada con ceros, de tamaño n x (n+1)
    augmented_matrix = [[0 for j in range(n + 1)] for i in range(n)]

    # Copiamos los elementos de la matriz A en la matriz aumentada
    for i in range(n):
        for j in range(n):
            augmented_matrix[i][j] = A[i][j]

    # Restamos el valor propio (eigenvalue) de los elementos de la diagonal principal
    # Esto corresponde a calcular A - lambda I, donde lambda es el valor propio
    for i in range(n):
        augmented_matrix[i][i] -= eigenvalue

    # Usamos eliminación gaussiana para resolver el sistema (A - lambda I)v = 0,
    v = gauss_elimination(augmented_matrix)

    # Normalizamos el vector propio para que tenga una norma unitaria.
    v_norm = np.array(v) / np.linalg.norm(v)

    if show:
        print("Para la siguiente matriz")
        for row in A:
            print(*row)

        print("El vector propio asociado al valor propio", np.round(eigenvalue, 2), "es", v_norm)
        print()

    # Retornamos el vector propio normalizado.
    return v_norm


def eigenvalue_power_method(A, x, mx=1000, tol=1e-9):
    """
    Computes the dominant eigenvalue and its corresponding eigenvector
    of a square matrix `A` using the Power Method.

    This method iteratively estimates the eigenvector associated with
    the largest (dominant) eigenvalue in magnitude, and refines the
    eigenvalue using the Rayleigh quotient.

    Args:
        A (numpy.ndarray): The square matrix for which the dominant eigenvalue
            and eigenvector are computed.
        x (numpy.ndarray): The initial guess for the eigenvector.
        mx (int, optional): The maximum number of iterations to perform. Default is 1000.
        tol (float, optional): The convergence tolerance. Iterations stop when the
            change in the eigenvector is less than this threshold. Default is 1e-9.

    Returns:
        tuple: A tuple `(eigenvalue, eigenvector)` where:
            - `eigenvalue` (float): The dominant eigenvalue of the matrix `A`.
            - `eigenvector` (numpy.ndarray): The corresponding normalized eigenvector.
    """

    x = x / np.linalg.norm(x)  # Normalize
    for _ in range(mx):
        Ax = np.dot(A, x)  # Compute Ax
        x_next = Ax / np.linalg.norm(Ax)  # Normalize
        eigenvalue = np.dot(x_next.T, np.dot(A, x_next))  # Use Rayleigh
        if np.linalg.norm(x_next - x) < tol:  # Check the tolerance
            break
        x = x_next  # Assign the next x to continue the iterations

    return eigenvalue, x_next


def eigenvalues_QR_algorithm(A: list[list[float]], mx=1000, tol=1e-6):
    """
    Computes the eigenvalues of a square matrix `A` using the QR algorithm.

    This function iteratively computes the QR decomposition of the matrix
    and updates `A` to converge towards an upper triangular matrix.
    The eigenvalues are extracted from the diagonal of the resulting matrix.

    Args:
        A (list[list[float]]): The input square matrix for which eigenvalues are computed.
        mx (int, optional): The maximum number of iterations to perform. Default is 1000.
        tol (float, optional): The convergence tolerance.

    Returns:
        list[float]: A sorted list of eigenvalues of the input matrix `A`.

    """
    A0 = A
    for _ in range(mx):
        Q, R = np.linalg.qr(A)  # Q es ortonormal y R es triangular superior
        A = np.dot(R, Q)

        # El error es la norma residual
        if np.linalg.norm(A - A0) < tol:
            break

        A0 = A

    eigenvalues = np.diag(A)
    return sorted(eigenvalues)
