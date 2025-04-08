import numpy as np

def get_selected_centroids(individual: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Seleciona os centroides com base em um indivíduo e um conjunto de candidatos.

    Args:
        individual (numpy.ndarray): Vetor binário representando a seleção de centroides 
                                    (1 para selecionado, 0 para não selecionado).
        candidates (numpy.ndarray): Array de candidatos a centroides.

    Returns:
        numpy.ndarray: Array contendo os centroides selecionados.
    """
    return candidates[individual == 1]

def generate_individual(n: int, k: int) -> np.ndarray:
    """
    Gera um indivíduo (vetor binário) com k bits ativos em n posições.

    Args:
        n (int): Número total de posições no vetor.
        k (int): Número de bits que devem ser ativos (valor 1).

    Returns:
        numpy.ndarray: Vetor binário representando o indivíduo.
    """
    bits = np.zeros(n, dtype=int)
    escolhidos = np.random.choice(n, size=k, replace=False)
    bits[escolhidos] = 1
    return bits

def generate_initial_population(n: int, k: int) -> np.ndarray:
    """
    Gera uma população inicial com um único indivíduo.  Os k últimos bits são ativados.

    Args:
        n (int): Tamanho do indivíduo (número de genes).
        k (int): Número de centroides (número de genes ativos).

    Returns:
        numpy.ndarray: Um indivíduo (vetor binário) representando a população inicial.
    """
     
    bits = np.zeros(n, dtype=int)
    bits[-k:] = 1  # Ativa os últimos k bits
    return bits