"""Rede neural artificial 4-8-3 para classificar o conjunto Iris.

A rede é implementada manualmente com NumPy. O Scikit-Learn é utilizado
somente para carregar os dados, separar treino/teste e padronizar as medidas.
"""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configurações do experimento
# ---------------------------------------------------------------------------
SEMENTE_ALEATORIA = 42
EPOCAS = 1000
TAXA_APRENDIZADO = 0.10
NUMERO_ENTRADAS = 4
NUMERO_NEURONIOS_OCULTOS = 8
NUMERO_CLASSES = 3

PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

CORES_ESPECIES = ["#2878B5", "#E69F00", "#26966F"]


# ---------------------------------------------------------------------------
# Funções matemáticas da rede neural
# ---------------------------------------------------------------------------
def relu(valores):
    """ReLU: mantém valores positivos e transforma os negativos em zero."""
    return np.maximum(0, valores)


def derivada_relu(valores):
    """Derivada da ReLU, utilizada durante a retropropagação."""
    return (valores > 0).astype(float)


def softmax(valores):
    """Transforma as saídas em probabilidades que somam 1 por amostra."""
    valores_estaveis = valores - np.max(valores, axis=1, keepdims=True)
    exponenciais = np.exp(valores_estaveis)
    return exponenciais / np.sum(exponenciais, axis=1, keepdims=True)


def entropia_cruzada(probabilidades, respostas_corretas):
    """Calcula a Cross-Entropy Loss para um conjunto de exemplos."""
    limite = 1e-12
    probabilidades_seguras = np.clip(probabilidades, limite, 1.0)
    quantidade = respostas_corretas.shape[0]
    return -np.sum(respostas_corretas * np.log(probabilidades_seguras)) / quantidade


class RedeNeuralIris:
    """Rede neural 4-8-3 treinada com backpropagation e gradiente descendente."""

    def __init__(self, semente=SEMENTE_ALEATORIA):
        gerador = np.random.default_rng(semente)

        # Inicialização He na camada ReLU e Xavier na camada de saída.
        self.pesos_entrada_oculta = gerador.normal(
            0,
            np.sqrt(2 / NUMERO_ENTRADAS),
            (NUMERO_ENTRADAS, NUMERO_NEURONIOS_OCULTOS),
        )
        # Os vieses também começam com valores aleatórios pequenos, como pede o enunciado.
        self.vieses_oculta = gerador.normal(
            0, 0.01, (1, NUMERO_NEURONIOS_OCULTOS)
        )

        self.pesos_oculta_saida = gerador.normal(
            0,
            np.sqrt(1 / NUMERO_NEURONIOS_OCULTOS),
            (NUMERO_NEURONIOS_OCULTOS, NUMERO_CLASSES),
        )
        self.vieses_saida = gerador.normal(0, 0.01, (1, NUMERO_CLASSES))

        self.historico_loss = []
        self.historico_acuracia = []

    def forward_pass(self, entradas):
        """Percorre a rede da entrada até as probabilidades de cada espécie."""
        soma_oculta = entradas @ self.pesos_entrada_oculta + self.vieses_oculta
        ativacao_oculta = relu(soma_oculta)

        soma_saida = ativacao_oculta @ self.pesos_oculta_saida + self.vieses_saida
        probabilidades = softmax(soma_saida)

        # Os valores intermediários são guardados para o backpropagation.
        cache = (entradas, soma_oculta, ativacao_oculta)
        return probabilidades, cache

    def backpropagation(self, probabilidades, respostas_corretas, cache):
        """Calcula os gradientes dos pesos e vieses pela regra da cadeia."""
        entradas, soma_oculta, ativacao_oculta = cache
        quantidade = entradas.shape[0]

        # Gradiente da combinação Softmax + Cross-Entropy.
        erro_saida = (probabilidades - respostas_corretas) / quantidade
        gradiente_pesos_saida = ativacao_oculta.T @ erro_saida
        gradiente_vieses_saida = np.sum(erro_saida, axis=0, keepdims=True)

        erro_oculta = erro_saida @ self.pesos_oculta_saida.T
        erro_oculta *= derivada_relu(soma_oculta)
        gradiente_pesos_oculta = entradas.T @ erro_oculta
        gradiente_vieses_oculta = np.sum(erro_oculta, axis=0, keepdims=True)

        return (
            gradiente_pesos_oculta,
            gradiente_vieses_oculta,
            gradiente_pesos_saida,
            gradiente_vieses_saida,
        )

    def atualizar_parametros(self, gradientes, taxa_aprendizado):
        """Aplica o gradiente descendente aos pesos e aos vieses."""
        (grad_w1, grad_b1, grad_w2, grad_b2) = gradientes

        self.pesos_entrada_oculta -= taxa_aprendizado * grad_w1
        self.vieses_oculta -= taxa_aprendizado * grad_b1
        self.pesos_oculta_saida -= taxa_aprendizado * grad_w2
        self.vieses_saida -= taxa_aprendizado * grad_b2

    def prever_probabilidades(self, entradas):
        """Retorna a probabilidade prevista para cada uma das três espécies."""
        probabilidades, _ = self.forward_pass(entradas)
        return probabilidades

    def treinar(self, entradas, respostas_corretas, epocas, taxa_aprendizado):
        """Treina a rede e registra loss e acurácia somente no conjunto de treino."""
        for epoca in range(epocas):
            probabilidades, cache = self.forward_pass(entradas)
            gradientes = self.backpropagation(
                probabilidades,
                respostas_corretas,
                cache,
            )
            self.atualizar_parametros(gradientes, taxa_aprendizado)

            # Recalcula após a atualização para registrar a situação da época.
            previsoes_atualizadas = self.prever_probabilidades(entradas)
            loss = entropia_cruzada(previsoes_atualizadas, respostas_corretas)
            acuracia = np.mean(
                np.argmax(previsoes_atualizadas, axis=1)
                == np.argmax(respostas_corretas, axis=1)
            )

            self.historico_loss.append(float(loss))
            self.historico_acuracia.append(float(acuracia))

            if (epoca + 1) % 200 == 0:
                print(
                    f"Época {epoca + 1:4d}/{epocas} | "
                    f"Loss: {loss:.6f} | "
                    f"Acurácia de treino: {acuracia * 100:.2f}%"
                )


# ---------------------------------------------------------------------------
# Métricas e gravação dos resultados
# ---------------------------------------------------------------------------
def calcular_matriz_confusao(respostas, previsoes, quantidade_classes):
    """Monta a matriz sem depender de um classificador pronto do Scikit-Learn."""
    matriz = np.zeros((quantidade_classes, quantidade_classes), dtype=int)
    for resposta, previsao in zip(respostas, previsoes):
        matriz[int(resposta), int(previsao)] += 1
    return matriz


def salvar_previsoes_csv(
    caminho,
    indices_teste,
    medidas_teste,
    nomes_corretos,
    nomes_previstos,
    probabilidades,
):
    """Salva as 30 previsões, as medidas originais e as probabilidades da rede."""
    cabecalho = [
        "numero_teste",
        "amostra_original",
        "comprimento_sepala_cm",
        "largura_sepala_cm",
        "comprimento_petala_cm",
        "largura_petala_cm",
        "especie_correta",
        "previsao_da_ia",
        "resultado",
        "confianca_da_previsao",
        "probabilidade_setosa",
        "probabilidade_versicolor",
        "probabilidade_virginica",
    ]

    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(cabecalho)

        for numero, (indice, medidas, correta, prevista, probabilidades_amostra) in enumerate(
            zip(
                indices_teste,
                medidas_teste,
                nomes_corretos,
                nomes_previstos,
                probabilidades,
            ),
            start=1,
        ):
            resultado = "Acertou" if correta == prevista else "Errou"
            confianca = float(np.max(probabilidades_amostra))
            escritor.writerow(
                [
                    numero,
                    int(indice) + 1,
                    *[f"{valor:.2f}" for valor in medidas],
                    correta,
                    prevista,
                    resultado,
                    f"{confianca:.6f}",
                    *[f"{valor:.6f}" for valor in probabilidades_amostra],
                ]
            )


def plotar_evolucao_loss(historico, caminho):
    figura, eixo = plt.subplots(figsize=(9, 5.2))
    eixo.plot(range(1, len(historico) + 1), historico, color="#2878B5", linewidth=2)
    eixo.set_title("Evolução do erro durante o treinamento", fontsize=14, weight="bold")
    eixo.set_xlabel("Épocas")
    eixo.set_ylabel("Cross-Entropy Loss")
    eixo.grid(alpha=0.25)
    figura.tight_layout()
    figura.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close(figura)


def plotar_evolucao_acuracia(historico, caminho):
    figura, eixo = plt.subplots(figsize=(9, 5.2))
    valores = np.array(historico) * 100
    eixo.plot(range(1, len(valores) + 1), valores, color="#26966F", linewidth=2)
    eixo.set_title("Evolução dos acertos no treinamento", fontsize=14, weight="bold")
    eixo.set_xlabel("Épocas")
    eixo.set_ylabel("Acurácia de treinamento (%)")
    eixo.set_ylim(0, 105)
    eixo.grid(alpha=0.25)
    figura.tight_layout()
    figura.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close(figura)


def plotar_matriz_confusao(matriz, nomes_classes, caminho):
    figura, eixo = plt.subplots(figsize=(7.2, 6))
    imagem = eixo.imshow(matriz, cmap="Blues")
    figura.colorbar(imagem, ax=eixo, fraction=0.046, pad=0.04, label="Quantidade de flores")

    eixo.set_title("Matriz de confusão — conjunto de teste", fontsize=14, weight="bold")
    eixo.set_xlabel("Espécie prevista pela rede")
    eixo.set_ylabel("Espécie correta")
    eixo.set_xticks(range(len(nomes_classes),), nomes_classes)
    eixo.set_yticks(range(len(nomes_classes)), nomes_classes)

    limite_cor = matriz.max() / 2 if matriz.size else 0
    for linha in range(matriz.shape[0]):
        for coluna in range(matriz.shape[1]):
            eixo.text(
                coluna,
                linha,
                str(matriz[linha, coluna]),
                ha="center",
                va="center",
                color="white" if matriz[linha, coluna] > limite_cor else "#202020",
                fontsize=13,
                weight="bold",
            )

    figura.tight_layout()
    figura.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close(figura)


def plotar_distribuicao_iris(dados_iris, caminho, eixo=None):
    """Mostra todas as amostras Iris usando comprimento e largura das pétalas."""
    figura_propria = eixo is None
    if figura_propria:
        figura, eixo = plt.subplots(figsize=(8.5, 5.8))

    for classe, nome in enumerate(dados_iris.target_names):
        mascara = dados_iris.target == classe
        eixo.scatter(
            dados_iris.data[mascara, 2],
            dados_iris.data[mascara, 3],
            label=nome.capitalize(),
            color=CORES_ESPECIES[classe],
            s=48,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.5,
        )

    eixo.set_title("Distribuição das espécies pelas pétalas", fontsize=14, weight="bold")
    eixo.set_xlabel("Comprimento da pétala (cm)")
    eixo.set_ylabel("Largura da pétala (cm)")
    eixo.legend(title="Espécie")
    eixo.grid(alpha=0.22)

    if figura_propria:
        figura.tight_layout()
        figura.savefig(caminho, dpi=180, bbox_inches="tight")
        plt.close(figura)


def criar_painel(dados_iris, historico_loss, historico_acuracia, matriz, metricas, caminho, distribuicao_treino, distribuicao_teste, erro_detalhado):
    """Gera uma imagem-resumo que explica o experimento em linguagem simples."""
    figura = plt.figure(figsize=(14, 8.5), facecolor="#F4F7FB")
    figura.text(
        0.5, 0.955, "REDE NEURAL ARTIFICIAL — IRIS",
        ha="center", fontsize=22, weight="bold", color="#18324B",
    )
    figura.text(
        0.5, 0.918,
        "Relatório da execução | Aprendizado supervisionado | Rede 4–8–3 com NumPy",
        ha="center", fontsize=11, color="#526477",
    )

    cartoes = [
        ("150", "flores no conjunto"),
        ("120", "usadas para aprender"),
        ("30", "reservadas para testar"),
        (f"{metricas['acuracia_teste'] * 100:.2f}%", "acerto no teste"),
    ]
    posicoes = [0.08, 0.30, 0.52, 0.74]
    for (valor, legenda), x in zip(cartoes, posicoes):
        figura.text(
            x, 0.82, valor, ha="left", fontsize=23, weight="bold", color="#176B87",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E5F0F5", "edgecolor": "#D2E1E9"},
        )
        figura.text(x, 0.775, legenda, ha="left", fontsize=10, color="#526477")

    # Painel esquerdo: o que foi feito com os dados.
    figura.text(0.075, 0.69, "COMO OS DADOS FORAM DIVIDIDOS", fontsize=12, weight="bold", color="#18324B")
    figura.text(
        0.075, 0.635,
        "Treinamento: 120 flores (80%)\n"
        "A rede conhece as respostas corretas e ajusta seus pesos.\n\n"
        "Teste: 30 flores (20%)\n"
        "Essas flores ficam separadas durante o aprendizado.\n"
        "Só são usadas depois para conferir se a rede aprendeu.",
        fontsize=10.5, color="#34495E", linespacing=1.5, va="top",
    )

    figura.text(0.53, 0.69, "O QUE A REDE RECEBE E FAZ", fontsize=12, weight="bold", color="#18324B")
    figura.text(
        0.53, 0.635,
        "1. Recebe 4 medidas em centímetros: sépala e pétala.\n"
        "2. Processa as medidas em 8 neurônios internos.\n"
        "3. Calcula 3 probabilidades, uma para cada espécie.\n"
        "4. Compara a previsão com a espécie correta no teste.\n\n"
        f"Treinamento: {metricas['epocas']} épocas | Taxa de aprendizado: {metricas['taxa_aprendizado']:.2f}",
        fontsize=10.5, color="#34495E", linespacing=1.5, va="top",
    )

    figura.text(0.075, 0.365, "RESULTADO DA AVALIAÇÃO", fontsize=12, weight="bold", color="#18324B")
    figura.text(
        0.075, 0.31,
        f"{metricas['acertos']} flores foram classificadas corretamente.\n"
        f"{metricas['erros']} flor(es) foram classificadas incorretamente.\n"
        f"Loss no teste: {metricas['loss_teste']:.6f} (medida do erro das probabilidades).",
        fontsize=10.5, color="#34495E", linespacing=1.55, va="top",
    )

    figura.text(0.53, 0.365, "DIVISÃO POR ESPÉCIE", fontsize=12, weight="bold", color="#18324B")
    linhas = []
    for nome in dados_iris.target_names:
        titulo = nome.capitalize()
        linhas.append(f"{titulo:<13} Treinamento: {distribuicao_treino[titulo]:>2}  |  Teste: {distribuicao_teste[titulo]:>2}")
    figura.text(0.53, 0.31, "\n".join(linhas), fontsize=10.5, family="monospace", color="#34495E", va="top", linespacing=1.6)

    figura.text(0.075, 0.145, "DETALHE DO ERRO", fontsize=11, weight="bold", color="#A33B32")
    figura.text(
        0.075, 0.105,
        erro_detalhado if erro_detalhado else "Nenhum erro ocorreu nesta execução.",
        fontsize=10, color="#34495E", va="top",
    )
    figura.text(
        0.5, 0.035,
        "Os gráficos de evolução e a matriz de confusão estão disponíveis separadamente na pasta resultados.",
        ha="center", fontsize=9, color="#65788A",
    )
    figura.savefig(caminho, dpi=180, bbox_inches="tight", facecolor=figura.get_facecolor())
    return figura


def salvar_metricas_json(caminho, metricas, nomes_classes, quantidade_treino, distribuicao_treino, distribuicao_teste):
    """Grava os parâmetros e resultados da execução em formato JSON."""
    dados = {
        "configuracao": {
            "arquitetura": "4-8-3",
            "epocas": EPOCAS,
            "taxa_aprendizado": TAXA_APRENDIZADO,
            "semente_aleatoria": SEMENTE_ALEATORIA,
            "inicializacao_pesos": "aleatoria (He e Xavier)",
            "inicializacao_vieses": "aleatoria normal, media 0 e desvio padrao 0.01",
            "divisao_treino_teste": "80/20 estratificada",
        },
        "dados": {
            "total": quantidade_treino + metricas["total_teste"],
            "treinamento": quantidade_treino,
            "teste": metricas["total_teste"],
            "classes": nomes_classes,
            "distribuicao_treino_por_classe": distribuicao_treino,
            "distribuicao_teste_por_classe": distribuicao_teste,
            "medidas_de_entrada": [
                "Comprimento da sepala (cm)",
                "Largura da sepala (cm)",
                "Comprimento da petala (cm)",
                "Largura da petala (cm)",
            ],
        },
        "resultados": metricas,
    }
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Janela explicativa para apresentar o experimento
# ---------------------------------------------------------------------------
def abrir_janela_resultados(
    dados_iris,
    metricas,
    indices_teste,
    medidas_teste,
    classes_teste,
    probabilidades_teste,
    historico_loss,
    historico_acuracia,
    matriz,
    distribuicao_treino,
    distribuicao_teste,
):
    """Abre uma interface com explicações, resultados e testes individuais."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        janela = tk.Tk()
    except Exception as erro:
        print(f"Janela gráfica indisponível neste ambiente: {erro}")
        print("Os resultados continuam salvos na pasta 'resultados'.")
        return

    janela.title("Rede Neural Iris — Entenda o experimento")
    janela.geometry("1180x820")
    janela.minsize(980, 700)
    janela.configure(bg="#F3F6FA")

    estilo = ttk.Style(janela)
    try:
        estilo.theme_use("clam")
    except tk.TclError:
        pass

    cor_fundo = "#F3F6FA"
    cor_texto = "#203247"
    cor_secundaria = "#52677D"
    cor_azul = "#176B87"
    cor_verde = "#237A57"
    cor_vermelho = "#A33B32"
    nomes_corretos = dados_iris.target_names[classes_teste]
    nomes_previstos = dados_iris.target_names[np.argmax(probabilidades_teste, axis=1)]
    posicoes_com_erro = np.where(nomes_corretos != nomes_previstos)[0]
    if len(posicoes_com_erro):
        posicao_erro = int(posicoes_com_erro[0])
        resumo_erro = (
            f"Único erro: registro Iris #{int(indices_teste[posicao_erro]) + 1} — "
            f"correta: {nomes_corretos[posicao_erro].capitalize()}; "
            f"previsão: {nomes_previstos[posicao_erro].capitalize()}."
        )
    else:
        resumo_erro = "Nenhuma classificação incorreta nesta execução."

    estilo.configure("TFrame", background=cor_fundo)
    estilo.configure("TLabel", background=cor_fundo, foreground=cor_texto, font=("Segoe UI", 10))
    estilo.configure("Titulo.TLabel", font=("Segoe UI", 20, "bold"), foreground="#18324B")
    estilo.configure("Subtitulo.TLabel", font=("Segoe UI", 10), foreground=cor_secundaria)
    estilo.configure("Secao.TLabel", font=("Segoe UI", 13, "bold"), foreground="#18324B")
    estilo.configure("Card.TFrame", background="white", relief="solid", borderwidth=1)
    estilo.configure("CardValor.TLabel", background="white", foreground=cor_azul, font=("Segoe UI", 22, "bold"))
    estilo.configure("CardLegenda.TLabel", background="white", foreground=cor_secundaria, font=("Segoe UI", 9))
    estilo.configure("TextoCard.TLabel", background="white", foreground=cor_texto, font=("Segoe UI", 10))
    estilo.configure("TNotebook", background=cor_fundo, borderwidth=0)
    estilo.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 10, "bold"))
    estilo.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", cor_azul)])
    estilo.configure("Treeview", font=("Segoe UI", 9), rowheight=27, background="white", fieldbackground="white")
    estilo.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
    estilo.configure("TButton", font=("Segoe UI", 9), padding=(10, 6))

    cabecalho = ttk.Frame(janela, padding=(24, 18, 24, 12))
    cabecalho.pack(fill="x")
    ttk.Label(cabecalho, text="Rede Neural Artificial — Iris", style="Titulo.TLabel").pack(anchor="w")
    ttk.Label(
        cabecalho,
        text="Entenda o que a inteligência artificial recebeu, como aprendeu e como foi avaliada.",
        style="Subtitulo.TLabel",
    ).pack(anchor="w", pady=(4, 0))

    abas = ttk.Notebook(janela)
    abas.pack(fill="both", expand=True, padx=18, pady=(0, 18))

    # Aba 1: resumo que explica o experimento antes de mostrar os gráficos.
    aba_visao = ttk.Frame(abas, padding=18)
    abas.add(aba_visao, text="Visão geral")

    ttk.Label(aba_visao, text="O que foi feito nesta execução?", style="Secao.TLabel").pack(anchor="w")
    ttk.Label(
        aba_visao,
        text=(
            "A rede recebeu 150 registros de flores. Cada registro contém quatro medidas em centímetros. "
            "Ela usou 120 flores para aprender e deixou outras 30 separadas para testar se conseguiu reconhecer padrões. "
            "Não são fotografias: a classificação usa somente números."
        ),
        wraplength=1080,
        justify="left",
    ).pack(anchor="w", pady=(6, 14))

    faixa_cartoes = ttk.Frame(aba_visao)
    faixa_cartoes.pack(fill="x", pady=(0, 16))

    def criar_cartao(pai, valor, legenda, coluna):
        cartao = ttk.Frame(pai, style="Card.TFrame", padding=(14, 12))
        cartao.grid(row=0, column=coluna, sticky="nsew", padx=(0 if coluna == 0 else 8, 0))
        ttk.Label(cartao, text=valor, style="CardValor.TLabel").pack(anchor="w")
        ttk.Label(cartao, text=legenda, style="CardLegenda.TLabel", wraplength=190).pack(anchor="w", pady=(2, 0))
        pai.grid_columnconfigure(coluna, weight=1)

    criar_cartao(faixa_cartoes, "150", "flores disponíveis no conjunto", 0)
    criar_cartao(faixa_cartoes, "120", "flores usadas para treinamento (80%)", 1)
    criar_cartao(faixa_cartoes, "30", "flores reservadas para teste (20%)", 2)
    criar_cartao(faixa_cartoes, f"{metricas['acuracia_teste'] * 100:.2f}%", "acerto obtido nas flores de teste", 3)

    colunas_info = ttk.Frame(aba_visao)
    colunas_info.pack(fill="both", expand=True)
    colunas_info.columnconfigure(0, weight=1)
    colunas_info.columnconfigure(1, weight=1)

    quadro_treino = ttk.Frame(colunas_info, style="Card.TFrame", padding=14)
    quadro_treino.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
    ttk.Label(quadro_treino, text="Treinamento: a fase de aprendizagem", style="Secao.TLabel").pack(anchor="w")
    ttk.Label(
        quadro_treino,
        text=(
            "As 120 flores de treinamento já têm a espécie correta conhecida. A rede faz previsões, "
            "compara suas respostas com as respostas corretas e ajusta os pesos e vieses para reduzir o erro.\n\n"
            f"Esse ciclo foi repetido {metricas['epocas']:,} vezes (épocas). "
            "Uma época é uma rodada completa de aprendizagem usando as 120 flores."
        ),
        style="TextoCard.TLabel", wraplength=490, justify="left",
    ).pack(anchor="w", pady=(8, 0))

    quadro_teste = ttk.Frame(colunas_info, style="Card.TFrame", padding=14)
    quadro_teste.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
    ttk.Label(quadro_teste, text="Teste: a fase de verificação", style="Secao.TLabel").pack(anchor="w")
    ttk.Label(
        quadro_teste,
        text=(
            "As 30 flores de teste não são usadas para ajustar a rede. Depois que o treinamento termina, "
            "a rede recebe essas medidas e tenta prever a espécie. Só então comparamos a previsão com "
            "a classificação verdadeira, que já existe no conjunto de dados.\n\n"
            "É por isso que o teste ajuda a avaliar se a rede aprendeu padrões, em vez de apenas decorar "
            "os exemplos usados para aprender."
        ),
        style="TextoCard.TLabel", wraplength=490, justify="left",
    ).pack(anchor="w", pady=(8, 0))

    quadro_resultado = ttk.Frame(colunas_info, style="Card.TFrame", padding=14)
    quadro_resultado.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
    ttk.Label(quadro_resultado, text="Resultado do teste", style="Secao.TLabel").pack(anchor="w")
    ttk.Label(
        quadro_resultado,
        text=(
            f"{metricas['acertos']} de {metricas['total_teste']} flores foram classificadas corretamente; "
            f"{metricas['erros']} tiveram previsão diferente da espécie correta.\n"
            f"Erro médio das probabilidades (Cross-Entropy Loss): {metricas['loss_teste']:.6f}.\n"
            f"Taxa de aprendizado usada no ajuste: {metricas['taxa_aprendizado']:.2f}.\n"
            f"{resumo_erro}"
        ),
        style="TextoCard.TLabel", wraplength=490, justify="left",
    ).pack(anchor="w", pady=(8, 0))

    quadro_divisao = ttk.Frame(colunas_info, style="Card.TFrame", padding=14)
    quadro_divisao.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
    ttk.Label(quadro_divisao, text="Como as espécies foram distribuídas", style="Secao.TLabel").pack(anchor="w")
    distribuicao_texto = "Espécie         Treinamento     Teste\n"
    distribuicao_texto += "──────────────────────────────────\n"
    for nome in dados_iris.target_names:
        especie = nome.capitalize()
        distribuicao_texto += f"{especie:<15} {distribuicao_treino[especie]:>6}          {distribuicao_teste[especie]:>3}\n"
    ttk.Label(
        quadro_divisao,
        text=distribuicao_texto,
        style="TextoCard.TLabel",
        font=("Consolas", 10),
        justify="left",
    ).pack(anchor="w", pady=(8, 0))

    # Aba 2: explicação dos termos e da arquitetura.
    aba_aprendizado = ttk.Frame(abas, padding=(10, 8))
    abas.add(aba_aprendizado, text="Como a IA aprende")
    area_rolagem = ttk.Frame(aba_aprendizado)
    area_rolagem.pack(fill="both", expand=True)
    tela_rolagem = tk.Canvas(area_rolagem, background=cor_fundo, highlightthickness=0)
    barra_aprendizado = ttk.Scrollbar(area_rolagem, orient="vertical", command=tela_rolagem.yview)
    tela_rolagem.configure(yscrollcommand=barra_aprendizado.set)
    barra_aprendizado.pack(side="right", fill="y")
    tela_rolagem.pack(side="left", fill="both", expand=True)
    conteudo_aprendizado = ttk.Frame(tela_rolagem, padding=10)
    janela_conteudo = tela_rolagem.create_window((0, 0), window=conteudo_aprendizado, anchor="nw")

    def atualizar_area_rolagem(event=None):
        tela_rolagem.configure(scrollregion=tela_rolagem.bbox("all"))

    def ajustar_largura_conteudo(event):
        tela_rolagem.itemconfigure(janela_conteudo, width=event.width)

    conteudo_aprendizado.bind("<Configure>", atualizar_area_rolagem)
    tela_rolagem.bind("<Configure>", ajustar_largura_conteudo)
    ttk.Label(conteudo_aprendizado, text="O caminho que a informação percorre", style="Secao.TLabel").pack(anchor="w")
    etapas = [
        ("1. Entrada", "A rede recebe quatro números: comprimento e largura da sépala, comprimento e largura da pétala."),
        ("2. Preparação", "As medidas são padronizadas usando média e desvio calculados somente com as 120 flores de treinamento."),
        ("3. Previsão", "A arquitetura 4–8–3 recebe as quatro medidas, processa oito neurônios internos e gera três probabilidades: Setosa, Versicolor e Virginica."),
        ("4. Comparação e erro", "Durante o treinamento, a Cross-Entropy Loss mede o quanto as probabilidades se afastam da resposta correta."),
        ("5. Ajuste", "O Backpropagation calcula como os pesos e vieses influenciam o erro. O Gradiente Descendente aplica os ajustes."),
        ("6. Repetição e teste", f"As etapas de aprendizagem são repetidas por {metricas['epocas']:,} épocas. Depois, a rede é avaliada nas 30 flores que ficaram separadas."),
    ]
    for titulo, explicacao in etapas:
        linha = ttk.Frame(conteudo_aprendizado, style="Card.TFrame", padding=(14, 10))
        linha.pack(fill="x", pady=5)
        ttk.Label(linha, text=titulo, font=("Segoe UI", 10, "bold"), background="white", foreground=cor_azul).pack(anchor="w")
        ttk.Label(linha, text=explicacao, style="TextoCard.TLabel", wraplength=1040, justify="left").pack(anchor="w", pady=(4, 0))

    ttk.Label(conteudo_aprendizado, text="O que significam os principais termos?", style="Secao.TLabel").pack(anchor="w", pady=(14, 6))
    termos = (
        "ReLU — função que mantém valores positivos e transforma valores negativos em zero na camada oculta.\n"
        "Softmax — transforma as três saídas em probabilidades que somam 100%.\n"
        "Época — uma rodada completa de aprendizagem usando todas as amostras de treinamento.\n"
        "Acurácia — porcentagem de previsões corretas. No resultado final, é calculada nas 30 flores de teste.\n"
        "Cross-Entropy Loss — medida do erro das probabilidades; em geral, valores menores indicam previsões melhores.\n"
        "Confiança — probabilidade que a rede atribuiu à espécie que escolheu. Uma confiança alta não garante que a previsão esteja correta."
    )
    ttk.Label(
        conteudo_aprendizado,
        text=termos,
        wraplength=1080,
        justify="left",
        background="white",
        padding=14,
    ).pack(fill="x")

    # Aba 3: tabela com todas as 30 flores reservadas para avaliação.
    aba_testes = ttk.Frame(abas, padding=18)
    abas.add(aba_testes, text="Flores testadas")
    ttk.Label(aba_testes, text="As 30 amostras que a rede teve que classificar", style="Secao.TLabel").pack(anchor="w")
    ttk.Label(
        aba_testes,
        text="Cada linha é uma flor que ficou fora do treinamento. Selecione uma linha para ver as medidas e as probabilidades calculadas pela rede.",
        wraplength=1080,
        justify="left",
    ).pack(anchor="w", pady=(5, 12))

    area_tabela = ttk.Frame(aba_testes)
    area_tabela.pack(fill="both", expand=True)
    colunas = ("numero", "registro", "correta", "prevista", "resultado", "confianca")
    tabela = ttk.Treeview(area_tabela, columns=colunas, show="headings", height=14, selectmode="browse")
    titulos = {
        "numero": "Teste",
        "registro": "Registro Iris",
        "correta": "Espécie correta",
        "prevista": "Previsão da IA",
        "resultado": "Resultado",
        "confianca": "Confiança da previsão",
    }
    larguras = {"numero": 65, "registro": 100, "correta": 145, "prevista": 145, "resultado": 100, "confianca": 160}
    for coluna in colunas:
        tabela.heading(coluna, text=titulos[coluna])
        tabela.column(coluna, width=larguras[coluna], anchor="center")
    barra_vertical = ttk.Scrollbar(area_tabela, orient="vertical", command=tabela.yview)
    barra_horizontal = ttk.Scrollbar(area_tabela, orient="horizontal", command=tabela.xview)
    tabela.configure(yscrollcommand=barra_vertical.set, xscrollcommand=barra_horizontal.set)
    tabela.grid(row=0, column=0, sticky="nsew")
    barra_vertical.grid(row=0, column=1, sticky="ns")
    barra_horizontal.grid(row=1, column=0, sticky="ew")
    area_tabela.rowconfigure(0, weight=1)
    area_tabela.columnconfigure(0, weight=1)
    tabela.tag_configure("acertou", foreground=cor_verde)
    tabela.tag_configure("errou", foreground=cor_vermelho)

    indices_itens = {}
    for numero, (indice, correta, probabilidades_amostra) in enumerate(
        zip(indices_teste, nomes_corretos, probabilidades_teste), start=1
    ):
        indice_previsto = int(np.argmax(probabilidades_amostra))
        prevista = dados_iris.target_names[indice_previsto]
        confianca = float(np.max(probabilidades_amostra)) * 100
        resultado = "Acertou" if correta == prevista else "Errou"
        item = tabela.insert(
            "", "end",
            values=(numero, f"#{int(indice) + 1}", correta.capitalize(), prevista.capitalize(), resultado, f"{confianca:.2f}%"),
            tags=("acertou" if resultado == "Acertou" else "errou",),
        )
        indices_itens[item] = numero - 1

    detalhe = ttk.Frame(aba_testes, style="Card.TFrame", padding=12)
    detalhe.pack(fill="x", pady=(12, 0))
    ttk.Label(detalhe, text="Detalhes da flor selecionada", style="Secao.TLabel").pack(anchor="w")
    texto_detalhe = tk.StringVar(value="Selecione uma das 30 linhas acima para consultar os dados dessa flor.")
    ttk.Label(detalhe, textvariable=texto_detalhe, style="TextoCard.TLabel", wraplength=1080, justify="left").pack(anchor="w", pady=(6, 0))

    def mostrar_detalhe(event=None):
        selecao = tabela.selection()
        if not selecao:
            return
        posicao = indices_itens[selecao[0]]
        medidas = medidas_teste[posicao]
        correta = dados_iris.target_names[int(classes_teste[posicao])].capitalize()
        probs = probabilidades_teste[posicao]
        previsao = dados_iris.target_names[int(np.argmax(probs))].capitalize()
        resultado = "ACERTO" if correta == previsao else "ERRO"
        registro = int(indices_teste[posicao]) + 1
        texto = (
            f"Registro original: #{registro}   |   Espécie correta: {correta}   |   Previsão da IA: {previsao}   |   Resultado: {resultado}\n"
            f"Medidas (cm): sépala — comprimento {medidas[0]:.1f}, largura {medidas[1]:.1f}; "
            f"pétala — comprimento {medidas[2]:.1f}, largura {medidas[3]:.1f}.\n"
            f"Probabilidades: Setosa {probs[0] * 100:.2f}%  |  Versicolor {probs[1] * 100:.2f}%  |  Virginica {probs[2] * 100:.2f}%."
        )
        texto_detalhe.set(texto)

    tabela.bind("<<TreeviewSelect>>", mostrar_detalhe)
    if tabela.get_children():
        primeiro = tabela.get_children()[0]
        tabela.selection_set(primeiro)
        mostrar_detalhe()

    # Aba 4: os gráficos ficam disponíveis como material complementar.
    aba_graficos = ttk.Frame(abas, padding=12)
    abas.add(aba_graficos, text="Gráficos de apoio")
    ttk.Label(
        aba_graficos,
        text="Os gráficos ajudam a observar o comportamento do treinamento e os resultados. As explicações principais estão nas outras abas.",
        wraplength=1080,
    ).pack(anchor="w", pady=(0, 6))

    figura = Figure(figsize=(10.4, 5.9), dpi=90, facecolor="white")
    eixo_loss = figura.add_subplot(221)
    eixo_acuracia = figura.add_subplot(222)
    eixo_matriz = figura.add_subplot(223)
    eixo_especies = figura.add_subplot(224)

    eixo_loss.plot(range(1, len(historico_loss) + 1), historico_loss, color="#2878B5", linewidth=1.8)
    eixo_loss.set_title("Erro durante o treinamento")
    eixo_loss.set_xlabel("Épocas")
    eixo_loss.set_ylabel("Cross-Entropy Loss")
    eixo_loss.grid(alpha=0.2)

    eixo_acuracia.plot(range(1, len(historico_acuracia) + 1), np.array(historico_acuracia) * 100, color="#26966F", linewidth=1.8)
    eixo_acuracia.set_title("Acertos durante o treinamento")
    eixo_acuracia.set_xlabel("Épocas")
    eixo_acuracia.set_ylabel("Acurácia (%)")
    eixo_acuracia.set_ylim(0, 105)
    eixo_acuracia.grid(alpha=0.2)

    imagem = eixo_matriz.imshow(matriz, cmap="Blues")
    nomes_classes = [nome.capitalize() for nome in dados_iris.target_names]
    eixo_matriz.set_title("Matriz de confusão — teste")
    eixo_matriz.set_xticks(range(3), nomes_classes)
    eixo_matriz.set_yticks(range(3), nomes_classes)
    eixo_matriz.set_xlabel("Previsão da IA")
    eixo_matriz.set_ylabel("Espécie correta")
    for linha in range(3):
        for coluna in range(3):
            eixo_matriz.text(coluna, linha, str(matriz[linha, coluna]), ha="center", va="center", color="#202020", weight="bold")

    for classe, nome in enumerate(dados_iris.target_names):
        mascara = dados_iris.target == classe
        eixo_especies.scatter(
            dados_iris.data[mascara, 2], dados_iris.data[mascara, 3],
            label=nome.capitalize(), color=CORES_ESPECIES[classe], s=20, alpha=0.75,
        )
    eixo_especies.set_title("Medidas das pétalas por espécie")
    eixo_especies.set_xlabel("Comprimento (cm)")
    eixo_especies.set_ylabel("Largura (cm)")
    eixo_especies.legend(fontsize=8)
    eixo_especies.grid(alpha=0.2)

    figura.tight_layout(pad=2.0)
    tela_graficos = FigureCanvasTkAgg(figura, master=aba_graficos)
    tela_graficos.draw()
    tela_graficos.get_tk_widget().pack(fill="both", expand=True)

    rodape = ttk.Frame(janela, padding=(18, 0, 18, 12))
    rodape.pack(fill="x")
    ttk.Label(
        rodape,
        text="Atividade Aula 10 — Inteligência Artificial | UNIPAC Barbacena",
        style="Subtitulo.TLabel",
    ).pack(side="left")
    ttk.Button(rodape, text="Fechar", command=janela.destroy).pack(side="right")

    janela.mainloop()


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------
def main():
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PROJETO: REDE NEURAL ARTIFICIAL — IRIS")
    print("=" * 65)
    print("Carregando os dados e preparando os conjuntos...\n")

    dados_iris = load_iris()
    entradas = dados_iris.data
    classes = dados_iris.target
    indices = np.arange(len(classes))

    # Separa os índices antes do pré-processamento para manter os dados originais.
    indices_treino, indices_teste = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEMENTE_ALEATORIA,
        stratify=classes,
    )

    entradas_treino_original = entradas[indices_treino]
    entradas_teste_original = entradas[indices_teste]
    classes_treino = classes[indices_treino]
    classes_teste = classes[indices_teste]

    # O padronizador aprende média e desvio somente no conjunto de treinamento.
    padronizador = StandardScaler()
    entradas_treino = padronizador.fit_transform(entradas_treino_original)
    entradas_teste = padronizador.transform(entradas_teste_original)

    # One-Hot Encoding: cada classe passa a ser um vetor com três posições.
    respostas_treino = np.eye(NUMERO_CLASSES)[classes_treino]

    print(f"Total de flores: {len(entradas)}")
    print(f"Amostras para treinamento: {len(entradas_treino)}")
    print(f"Amostras para teste: {len(entradas_teste)}")
    print("Arquitetura da rede: 4 entradas -> 8 neurônios -> 3 saídas")
    print(f"Épocas: {EPOCAS} | Taxa de aprendizado: {TAXA_APRENDIZADO}\n")

    rede = RedeNeuralIris()
    print("Iniciando o treinamento...\n")
    rede.treinar(
        entradas_treino,
        respostas_treino,
        EPOCAS,
        TAXA_APRENDIZADO,
    )

    # A avaliação abaixo acontece somente depois do treinamento.
    probabilidades_teste = rede.prever_probabilidades(entradas_teste)
    previsoes_teste = np.argmax(probabilidades_teste, axis=1)
    acuracia_teste = float(np.mean(previsoes_teste == classes_teste))
    loss_teste = float(
        entropia_cruzada(
            probabilidades_teste,
            np.eye(NUMERO_CLASSES)[classes_teste],
        )
    )
    matriz = calcular_matriz_confusao(classes_teste, previsoes_teste, NUMERO_CLASSES)

    acertos = int(np.sum(previsoes_teste == classes_teste))
    total_teste = int(len(classes_teste))
    erros = total_teste - acertos
    nomes_corretos = dados_iris.target_names[classes_teste]
    nomes_previstos = dados_iris.target_names[previsoes_teste]

    metricas = {
        "acuracia_teste": acuracia_teste,
        "loss_teste": loss_teste,
        "acertos": acertos,
        "erros": erros,
        "total_teste": total_teste,
        "epocas": EPOCAS,
        "taxa_aprendizado": TAXA_APRENDIZADO,
        "loss_final_treinamento": float(rede.historico_loss[-1]),
        "acuracia_final_treinamento": float(rede.historico_acuracia[-1]),
    }

    # Geração dos arquivos de resultados.
    plotar_evolucao_loss(
        rede.historico_loss,
        PASTA_RESULTADOS / "evolucao_loss.png",
    )
    plotar_evolucao_acuracia(
        rede.historico_acuracia,
        PASTA_RESULTADOS / "evolucao_acuracia.png",
    )
    plotar_matriz_confusao(
        matriz,
        [nome.capitalize() for nome in dados_iris.target_names],
        PASTA_RESULTADOS / "matriz_confusao.png",
    )
    plotar_distribuicao_iris(
        dados_iris,
        PASTA_RESULTADOS / "distribuicao_especies.png",
    )
    salvar_previsoes_csv(
        PASTA_RESULTADOS / "previsoes_teste.csv",
        indices_teste,
        entradas_teste_original,
        nomes_corretos,
        nomes_previstos,
        probabilidades_teste,
    )

    distribuicao_treino = {
        nome.capitalize(): int(np.sum(classes_treino == indice))
        for indice, nome in enumerate(dados_iris.target_names)
    }
    distribuicao_teste = {
        nome.capitalize(): int(np.sum(classes_teste == indice))
        for indice, nome in enumerate(dados_iris.target_names)
    }
    posicoes_erradas = np.where(previsoes_teste != classes_teste)[0]
    if len(posicoes_erradas):
        posicao_erro = int(posicoes_erradas[0])
        erro_detalhado = (
            f"Registro Iris #{int(indices_teste[posicao_erro]) + 1}: espécie correta "
            f"{nomes_corretos[posicao_erro].capitalize()}, mas a rede previu "
            f"{nomes_previstos[posicao_erro].capitalize()}."
        )
    else:
        erro_detalhado = "Nenhuma classificação incorreta foi registrada no conjunto de teste."

    painel = criar_painel(
        dados_iris,
        rede.historico_loss,
        rede.historico_acuracia,
        matriz,
        metricas,
        PASTA_RESULTADOS / "painel_aprendizado_iris.png",
        distribuicao_treino,
        distribuicao_teste,
        erro_detalhado,
    )
    salvar_metricas_json(
        PASTA_RESULTADOS / "metricas_execucao.json",
        metricas,
        [nome.capitalize() for nome in dados_iris.target_names],
        len(entradas_treino),
        {nome.capitalize(): int(np.sum(classes_treino == indice)) for indice, nome in enumerate(dados_iris.target_names)},
        {nome.capitalize(): int(np.sum(classes_teste == indice)) for indice, nome in enumerate(dados_iris.target_names)},
    )

    print("\n" + "=" * 65)
    print("RESULTADO DA AVALIAÇÃO")
    print("=" * 65)
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Acertos: {acertos} de {total_teste}")
    print(f"Erros: {erros}")
    print(f"Cross-Entropy Loss no teste: {loss_teste:.6f}")
    print(f"Loss final no treinamento: {rede.historico_loss[-1]:.6f}")
    print("\nMatriz de confusão (linhas = espécie correta; colunas = previsão):")
    print(matriz)

    print("\nAlgumas previsões reais do conjunto de teste:")
    print(f"{'Correta':<15} {'Previsão':<15} {'Resultado':<10}")
    print("-" * 42)
    for correta, prevista in zip(nomes_corretos[:10], nomes_previstos[:10]):
        resultado = "Acertou" if correta == prevista else "Errou"
        print(f"{correta.capitalize():<15} {prevista.capitalize():<15} {resultado:<10}")

    print(f"\nArquivos salvos em: {PASTA_RESULTADOS}")
    print("Abrindo a janela explicativa com visão geral, etapas de aprendizagem e lista dos 30 testes...")
    plt.close(painel)
    abrir_janela_resultados(
        dados_iris,
        metricas,
        indices_teste,
        entradas_teste_original,
        classes_teste,
        probabilidades_teste,
        rede.historico_loss,
        rede.historico_acuracia,
        matriz,
        distribuicao_treino,
        distribuicao_teste,
    )


if __name__ == "__main__":
    main()
