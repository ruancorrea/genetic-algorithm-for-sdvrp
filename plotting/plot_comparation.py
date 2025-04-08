import matplotlib.pyplot as plt

def plot(
        centroides_1, 
        centroides_2, 
        title_1="Centroides iniciais", 
        title_2="Centroides Otimizados", 
        title_3="Comparação dos Centroides"
    ):
    """
    Plota três conjuntos de centroides: iniciais, otimizados e comparação entre eles.
    Args:
        centroides_1: Lista de centroides do primeiro plot (iniciais).
        centroides_2: Lista de centroides do segundo plot (otimizados).
        title_1: Título do primeiro plot.
        title_2: Título do segundo plot.
        title_3: Título do terceiro plot.
    """

    # Converte os centroides para tuplas para facilitar a comparação
    centroides_1_tuples = [tuple(c) for c in centroides_1]
    centroides_2_tuples = [tuple(c) for c in centroides_2]

    # Encontra os centroides que estão apenas no primeiro plot (vermelho)
    centroides_apenas_1 = [c for c in centroides_1_tuples if c not in centroides_2_tuples]

    # Encontra os centroides que estão apenas no segundo plot (verde)
    centroides_apenas_2 = [c for c in centroides_2_tuples if c not in centroides_1_tuples]

    # Encontra os centroides que estão em ambos os plots (azul)
    centroides_em_ambos = [c for c in centroides_1_tuples if c in centroides_2_tuples]

    # Cria a figura com três subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Plota os centroides no primeiro subplot (iniciais)
    ax1.scatter([c[1] for c in centroides_1_tuples], [c[0] for c in centroides_1_tuples], marker='o', color='orange', s=50, label='Centroides')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(title_1)
    ax1.legend()
    ax1.grid(True)

    # Plota os centroides no segundo subplot (otimizados)
    ax2.scatter([c[1] for c in centroides_2_tuples], [c[0] for c in centroides_2_tuples], marker='o', color='blue', s=50, label='Centroides Otimizados')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(title_2)
    ax2.legend()
    ax2.grid(True)

    # Plota os centroides no terceiro subplot (comparação)
    ax3.scatter([c[1] for c in centroides_apenas_1], [c[0] for c in centroides_apenas_1], marker='x', color='red', s=50, label='Centroides Removidos')
    ax3.scatter([c[1] for c in centroides_apenas_2], [c[0] for c in centroides_apenas_2], marker='o', color='green', s=50, label='Centroides Adicionados')
    ax3.scatter([c[1] for c in centroides_em_ambos], [c[0] for c in centroides_em_ambos], marker='o', color='blue', s=50, label='Centroides Mantidos')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title(title_3)
    ax3.legend()
    ax3.grid(True)

    # Exibe os plots
    plt.show()

#plot(initial_centroids, best_centroids_selected)