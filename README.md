# Rede Neural Artificial — Classificação Iris

Projeto acadêmico da **Atividade Aula 10 — Inteligência Artificial**, desenvolvido para a UNIPAC Barbacena. A proposta é implementar uma rede neural artificial do zero para classificar as três espécies do conjunto de dados Iris.

A rede utiliza **NumPy** nos cálculos matemáticos e no treinamento. O Scikit-Learn é utilizado para carregar os dados, realizar a divisão estratificada e padronizar as medidas. O Matplotlib gera os gráficos de apoio. A janela principal é organizada com Tkinter, biblioteca incluída na instalação padrão do Python para Windows, e prioriza explicações e informações do experimento.

## Resultado desta execução

<details>
<summary>Ver métricas registradas na execução final</summary>

| Indicador | Resultado |
|---|---:|
| Exemplos no conjunto Iris | 150 |
| Exemplos de treinamento | 120 (80%) |
| Exemplos de teste | 30 (20%) |
| Arquitetura | 4–8–3 |
| Épocas | 1.000 |
| Taxa de aprendizado | 0,10 |
| Acurácia no teste | **96,67%** |
| Classificações corretas | 29 de 30 |
| Classificações incorretas | 1 |
| Cross-Entropy Loss no teste | 0,078004 |
| Loss final no treinamento | 0,049669 |
| Acurácia final no treinamento | 97,50% |

A matriz de confusão desta execução foi:

| Espécie correta / previsão | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|
| **Setosa** | 10 | 0 | 0 |
| **Versicolor** | 0 | 9 | 1 |
| **Virginica** | 0 | 0 | 10 |

Esses valores correspondem à execução registrada nos arquivos da pasta `resultados`. Eles não representam uma garantia de resultado em outras configurações.

</details>

## Janela gráfica: informações antes dos gráficos

Ao executar o programa no Windows, abre uma janela com quatro abas:

- **Visão geral:** explica o que a rede recebeu, mostra as 150 flores, a divisão entre 120 amostras de treinamento e 30 de teste, a distribuição por espécie e o resultado final.
- **Como a IA aprende:** descreve em linguagem simples a entrada, a preparação dos dados, a previsão, o cálculo do erro, os ajustes e a repetição do treinamento.
- **Flores testadas:** lista as 30 amostras reservadas para avaliação. Ao selecionar uma linha, são exibidas as quatro medidas da flor, a espécie correta, a previsão da rede e as probabilidades para as três espécies.
- **Gráficos de apoio:** reúne a evolução da loss, a evolução da acurácia no treinamento, a matriz de confusão e a distribuição das espécies. Os gráficos complementam as explicações, mas não são necessários para entender o fluxo do experimento.

A imagem abaixo é um resumo estático do experimento em formato SVG, para visualização nítida diretamente no GitHub. Os gráficos de apoio são gerados em PNG automaticamente na pasta `resultados` quando o programa é executado no Windows.

![Resumo explicativo da rede neural Iris](resultados/painel_aprendizado_iris.svg)

## Objetivo do projeto

O conjunto Iris contém 150 exemplos de flores, divididos em três espécies: Setosa, Versicolor e Virginica. Cada exemplo possui quatro medidas numéricas:

- comprimento da sépala;
- largura da sépala;
- comprimento da pétala;
- largura da pétala.

O programa recebe essas quatro medidas e prevê a espécie correspondente. Ele não analisa fotografias, cores ou imagens das flores.

## Rede neural e sistema especialista

Um sistema especialista utiliza regras previamente definidas, como condições do tipo SE/ENTÃO. Já esta rede neural aprende padrões a partir de exemplos rotulados: durante o treinamento, ajusta pesos e vieses para aproximar suas previsões das classes corretas.

Por isso, este projeto é uma implementação de **aprendizado supervisionado**, e não um sistema especialista.

## Arquitetura 4–8–3

A arquitetura contém:

1. **4 entradas:** recebem as quatro medidas da flor.
2. **8 neurônios na camada oculta:** processam as entradas utilizando a função ReLU.
3. **3 saídas:** representam as probabilidades de Setosa, Versicolor e Virginica, calculadas pela Softmax.

### Funções e etapas implementadas

- **ReLU:** transforma valores negativos em zero e mantém os positivos.
- **Softmax:** converte as saídas em probabilidades cuja soma é 1.
- **Forward Pass:** calcula as ativações e as probabilidades da rede.
- **Cross-Entropy Loss:** mede a diferença entre as probabilidades previstas e a classe correta.
- **Backpropagation:** calcula os gradientes dos pesos e vieses.
- **Gradiente Descendente:** atualiza os parâmetros para reduzir o erro.
- **One-Hot Encoding:** representa cada uma das três classes com um vetor de três posições.

O treinamento é feito em lote completo: em cada época, os gradientes são calculados usando todas as 120 amostras de treinamento.

## Preparação e avaliação dos dados

A divisão é estratificada e mantém a proporção das três espécies: 80% para treinamento e 20% para teste. O padronizador é ajustado exclusivamente com os dados de treinamento; em seguida, o mesmo ajuste é aplicado ao conjunto de teste. Dessa forma, o teste não participa do cálculo dos parâmetros de padronização.

A rede é treinada sem consultar o conjunto de teste. A avaliação final ocorre depois do treinamento e utiliza somente as 30 amostras reservadas. A matriz de confusão apresenta as classes verdadeiras nas linhas e as previsões nas colunas.

## Gráficos e arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `painel_aprendizado_iris.svg` | Resumo visual vetorial com explicação da divisão dos dados, etapas e indicadores. |
| `painel_aprendizado_iris.png` | Versão PNG gerada automaticamente na execução local. |
| `evolucao_loss.png` | Versão PNG gerada automaticamente na execução local. |
| `evolucao_acuracia.png` | Versão PNG gerada automaticamente na execução local. |
| `matriz_confusao.png` | Versão PNG gerada automaticamente na execução local. |
| `distribuicao_especies.png` | Versão PNG gerada automaticamente na execução local. |
| `previsoes_teste.csv` | As 30 amostras de teste, com as quatro medidas, espécie correta, previsão, resultado, confiança e três probabilidades. |
| `metricas_execucao.json` | Configuração e métricas registradas na execução. |

A curva de acurácia é calculada sobre os dados de treinamento. A matriz de confusão e as métricas finais são calculadas com os dados de teste. O gráfico de distribuição é apenas descritivo e apresenta as amostras do conjunto Iris.

## Tecnologias utilizadas

- Python
- NumPy
- Scikit-Learn
- Matplotlib
- Tkinter (biblioteca padrão do Python, usada na janela gráfica)

Não são utilizados TensorFlow, PyTorch ou classificadores prontos para treinar a rede.

## Instalação no Windows

Abra o PowerShell ou o terminal na pasta do projeto e instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Caso utilize o inicializador `py` do Windows:

```powershell
py -m pip install -r requirements.txt
```

## Como executar

Na pasta principal do projeto, execute:

```powershell
python rede_neural_iris.py
```

Ou, no Windows:

```powershell
py rede_neural_iris.py
```

O programa treina a rede, apresenta as métricas no terminal, gera os arquivos de resultado e abre a janela explicativa. No Windows, a janela permite consultar o resumo, entender as etapas, explorar as 30 flores de teste e acessar os gráficos em uma aba separada. Em ambientes sem interface gráfica, os arquivos continuam sendo gerados normalmente e o programa informa que não conseguiu abrir a janela.

## Estrutura de pastas

```text
projeto_rede_neural_iris/
├── .gitignore
├── rede_neural_iris.py
├── README.md
├── requirements.txt
├── docs/
│   └── explicacao_do_projeto.md
└── resultados/
    ├── painel_aprendizado_iris.svg
    ├── previsoes_teste.csv
    └── metricas_execucao.json
```

## Reprodutibilidade

A semente aleatória está fixa em `42`, e a divisão dos dados também utiliza `random_state=42`. Isso permite repetir o experimento com a mesma configuração e obter os mesmos resultados, considerando versões compatíveis das bibliotecas.

## Atividade acadêmica

**Instituição:** UNIPAC Barbacena  
**Disciplina:** Inteligência Artificial  
**Atividade:** Aula 10 — AECO  
**Professor:** Felipe

Para a explicação detalhada das etapas matemáticas, consulte [`docs/explicacao_do_projeto.md`](docs/explicacao_do_projeto.md).