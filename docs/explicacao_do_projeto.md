# Explicação técnica — Rede Neural Artificial Iris

## 1. Objetivo

O projeto implementa uma rede neural artificial para classificar flores do conjunto Iris em três espécies: Setosa, Versicolor e Virginica. A atividade utiliza aprendizado supervisionado: durante o treinamento, cada exemplo possui medidas e uma classe correta associada.

A rede foi construída manualmente com NumPy. O Scikit-Learn é usado para carregar o conjunto, separar os dados e padronizar as medidas. Não há uso de TensorFlow, PyTorch ou de um classificador pronto.

## 2. Rede neural e sistema especialista

Um sistema especialista depende de regras definidas previamente, como condições do tipo SE/ENTÃO. A rede neural deste projeto não recebe regras que determinam diretamente a espécie. Ela ajusta pesos e vieses durante o treinamento para aprender relações entre as medidas e as classes conhecidas.

## 3. Dados de entrada

Cada flor é representada por quatro medidas, em centímetros:

- comprimento da sépala;
- largura da sépala;
- comprimento da pétala;
- largura da pétala.

O conjunto contém 150 registros, divididos igualmente entre as três espécies. Não são utilizadas fotografias: a classificação é feita somente com as medidas numéricas.

### As três espécies

- **Setosa:** em geral, apresenta pétalas bem menores que as outras duas espécies no conjunto Iris.
- **Versicolor:** costuma apresentar medidas intermediárias.
- **Virginica:** geralmente apresenta pétalas maiores e pode ter medidas próximas às da Versicolor.

Essas descrições ajudam a interpretar os dados, mas não são regras fixas de classificação.

## 4. Preparação dos dados

A divisão utiliza 80% dos registros para treinamento e 20% para teste, com estratificação. A estratificação preserva a proporção das três espécies nos dois grupos.

O `StandardScaler` é ajustado somente com as amostras de treinamento. Depois, o mesmo ajuste é aplicado aos dados de treinamento e de teste. Assim, informações do teste não participam do cálculo de média e desvio usados na padronização.

As classes são convertidas para One-Hot Encoding. Por exemplo, as três classes são representadas por vetores de três posições, com um único valor 1 indicando a classe correta.

## 5. Arquitetura 4–8–3

A rede possui três partes:

1. **Entrada (4):** recebe as quatro medidas da flor.
2. **Camada oculta (8):** contém oito neurônios e utiliza ReLU.
3. **Saída (3):** produz uma probabilidade para cada espécie por meio da Softmax.

Os pesos representam a intensidade das conexões. Os vieses permitem ajustar a saída de cada neurônio. No início, os pesos são gerados aleatoriamente: utiliza-se inicialização He na camada ReLU e Xavier na camada de saída. Os dois conjuntos de vieses também são inicializados aleatoriamente, com distribuição normal de média zero e desvio padrão 0,01. A semente fixa permite repetir a mesma inicialização nas execuções.

## 6. Funções matemáticas

### ReLU

A ReLU mantém valores positivos e transforma valores negativos em zero. Ela é aplicada na camada oculta:

`ReLU(x) = max(0, x)`

### Softmax

A Softmax transforma os três valores de saída em probabilidades. Para cada flor, as três probabilidades somam aproximadamente 1 (100%). A maior probabilidade indica a espécie prevista pela rede.

### Cross-Entropy Loss

A Cross-Entropy Loss mede a diferença entre as probabilidades previstas e a classe correta. Valores menores indicam previsões melhores em relação às respostas esperadas.

### Fórmulas principais

Na camada oculta, a combinação linear é calculada por `Z1 = X · W1 + b1`, seguida da ativação `A1 = ReLU(Z1)`. Na saída, a rede calcula `Z2 = A1 · W2 + b2` e aplica a Softmax:

`Softmax(z_i) = exp(z_i) / soma(exp(z_j))`

Para uma amostra, a Cross-Entropy é `L = -soma(y_i · log(p_i))`, em que `y` é o vetor One-Hot da resposta correta e `p` contém as probabilidades previstas. O programa calcula a média dessa perda para o lote.

Na combinação Softmax + Cross-Entropy, o gradiente em relação aos valores de saída é `(P - Y) / n`, sendo `n` a quantidade de exemplos do lote. A retropropagação utiliza esse resultado e a derivada da ReLU para calcular os gradientes das duas camadas. Por fim, cada parâmetro é atualizado por `parâmetro = parâmetro - taxa_de_aprendizado · gradiente`.

## 7. Como ocorre o aprendizado

O treinamento é feito com as seguintes etapas:

1. **Forward Pass:** as medidas atravessam as camadas e a rede calcula as probabilidades.
2. **Cálculo do erro:** a Cross-Entropy compara as probabilidades com as classes corretas.
3. **Backpropagation:** a rede calcula os gradientes, indicando como cada peso e viés influencia o erro.
4. **Gradiente Descendente:** os parâmetros são atualizados na direção que reduz o erro, usando a taxa de aprendizado configurada.
5. **Repetição:** o processo é repetido durante 1.000 épocas.

Neste projeto, o treinamento utiliza gradiente descendente em lote completo: em cada época, os gradientes são calculados usando todas as amostras de treinamento.

## 8. Avaliação

O conjunto de teste fica separado durante todo o treinamento. Depois que os pesos são ajustados, a rede recebe as amostras de teste e suas previsões são comparadas com as classes verdadeiras.

A acurácia corresponde à proporção de previsões corretas. A matriz de confusão detalha os acertos e as confusões entre as espécies. A loss de teste mede o erro das probabilidades no conjunto reservado. Na execução registrada, a rede confundiu uma amostra de Versicolor com Virginica; as demais 29 amostras foram classificadas corretamente.

## 9. Janela gráfica e gráficos de apoio

A interface principal foi organizada para explicar o experimento antes de apresentar gráficos. Ela utiliza Tkinter, biblioteca padrão do Python, e possui quatro abas:

- **Visão geral:** apresenta o total de flores, a quantidade usada no treinamento, a quantidade separada para teste, a acurácia e a distribuição das espécies nos dois grupos. Também explica a diferença entre aprender e testar.
- **Como a IA aprende:** descreve as etapas do fluxo de dados e explica os termos técnicos em linguagem simples.
- **Flores testadas:** lista individualmente as 30 amostras de teste. Ao selecionar uma delas, é possível consultar as quatro medidas originais, a espécie correta, a previsão e as probabilidades calculadas para Setosa, Versicolor e Virginica.
- **Gráficos de apoio:** reúne os gráficos de evolução e distribuição, além da matriz de confusão. Eles são complementares à explicação textual.

Os gráficos disponíveis são:

- **Evolução da Loss:** mostra como a Cross-Entropy Loss do treinamento varia ao longo das épocas.
- **Evolução da Acurácia:** mostra a proporção de classificações corretas no conjunto de treinamento durante as épocas.
- **Matriz de Confusão:** compara as espécies verdadeiras e previstas exclusivamente no conjunto de teste. As linhas representam as classes corretas e as colunas, as previsões.
- **Distribuição das Espécies:** apresenta as 150 flores em um gráfico de comprimento e largura das pétalas. Esse gráfico é descritivo e ajuda a visualizar a sobreposição entre algumas classes.

## 10. Arquivos de resultado

A pasta `resultados` contém um resumo visual explicativo, os quatro gráficos separados, a tabela CSV com as 30 previsões e suas medidas e probabilidades, e um arquivo JSON com as métricas da execução.

Os números apresentados no README e no JSON são produzidos pela execução do programa. Eles não são garantias de desempenho em outros ambientes ou com outras configurações.
