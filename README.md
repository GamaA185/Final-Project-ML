# Machine Learning Hands-On - Final Project

## 1. Introdução

O Machine Learning, ou aprendizado de máquina, é uma área da computação e da ciência de dados voltada ao desenvolvimento de métodos capazes de identificar padrões em conjuntos de dados e realizar previsões a partir desses padrões. Entre as principais abordagens de aprendizado supervisionado estão os problemas de regressão, nos quais o objetivo é prever uma variável numérica contínua a partir de um conjunto de variáveis de entrada.

Neste projeto foi utilizado o dataset Diabetes, disponibilizado pela biblioteca Scikit-learn, com o objetivo de aplicar e comparar diferentes métodos de regressão. Foram estudados modelos de Regressão Linear, Ridge, Lasso, Regressão Polinomial, Decision Tree Regressor e Random Forest Regressor. Além disso, foram utilizadas técnicas de análise exploratória, validação cruzada e métricas de avaliação para analisar o comportamento dos modelos.

## 2. Descrição do dataset

O conjunto de dados utilizado foi obtido por meio da função `load_diabetes()` da biblioteca Scikit-learn. Após o carregamento, os dados foram armazenados em um DataFrame do Pandas.

O dataset apresenta 442 observações e 11 colunas, sendo 10 variáveis utilizadas como características de entrada e uma variável denominada `target`, utilizada como variável-alvo. As características são `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5` e `s6`.

As variáveis `age`, `sex`, `bmi` e `bp` representam, respectivamente, idade, sexo, índice de massa corporal e pressão sanguínea. As variáveis `s1` a `s6` correspondem a diferentes medidas bioquímicas presentes no conjunto de dados.

A variável `target` representa a variável contínua que o modelo deve prever, relacionada à progressão da doença. Como seu valor é numérico e contínuo, o problema foi tratado como uma tarefa de regressão.

A análise inicial dos dados foi realizada utilizando as dimensões do DataFrame, os tipos de dados, a quantidade de valores ausentes, a quantidade de registros duplicados e as estatísticas descritivas. O dataset utilizado no projeto possui 442 observações e, conforme verificado na análise exploratória, não foram identificados valores ausentes ou registros duplicados.

## 3. Análise estatística e exploratória

Inicialmente foram implementadas funções próprias para o cálculo de média, mediana, moda, variância e desvio padrão. Essas funções foram aplicadas à variável `target`.

A análise estatística foi complementada pela utilização do método `describe()` do Pandas, permitindo observar medidas como média, desvio padrão, valores mínimo e máximo e quartis das variáveis.

Também foi construída a distribuição da variável-alvo por meio de um histograma. Essa visualização permite observar a distribuição dos valores de progressão da doença e verificar sua dispersão.

Foram produzidos ainda gráficos de dispersão relacionando cada uma das variáveis de entrada à variável `target`. Foram analisadas as relações entre `target` e `bmi`, `age`, `sex`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5` e `s6`.

A análise exploratória foi complementada por uma matriz de correlação. Essa matriz permite avaliar a associação linear entre as variáveis e identificar quais características apresentam maior relação linear com a variável-alvo.

Entre as variáveis analisadas, destaca-se a relação entre `bmi` e `target`, que apresenta uma associação linear positiva mais evidente do que algumas das demais variáveis. Essa observação é coerente com a importância atribuída posteriormente à variável `bmi` pelo modelo Random Forest.

## 4. Preparação dos dados

Para a construção dos modelos, a variável `target` foi separada das demais variáveis. O conjunto de características foi armazenado em `X`, enquanto a variável-alvo foi armazenada em `y`.

Os dados foram divididos em dois conjuntos utilizando a função `train_test_split()`. Foi utilizada uma proporção de 80% dos dados para treinamento e 20% para teste, com `random_state=42`, permitindo reproduzir a mesma divisão dos dados.

Dessa forma, foram utilizados 353 registros no conjunto de treinamento e 89 registros no conjunto de teste.

O conjunto de treinamento foi utilizado para ajustar os modelos, enquanto o conjunto de teste foi mantido separado para avaliar o desempenho dos modelos em dados que não participaram diretamente do treinamento.

## 5. Regressão Linear

A Regressão Linear foi utilizada como um dos modelos principais do projeto. Esse método procura estabelecer uma relação aproximadamente linear entre as variáveis de entrada e a variável-alvo.

O modelo foi inicialmente avaliado por meio de uma validação cruzada com seis folds. Em cada etapa, parte dos dados de treinamento foi utilizada para treinamento e outra parte para validação.

Os resultados médios obtidos na validação cruzada foram:

| Métrica | Resultado médio |
| ------- | --------------: |
| MAE     |         44,7702 |
| MSE     |       3047,5143 |
| RMSE    |         55,0825 |
| R²      |          0,4814 |

Após a validação cruzada, foi ajustado um modelo final utilizando todo o conjunto de treinamento. No conjunto de teste, a Regressão Linear apresentou:

| Métrica | Resultado |
| ------- | --------: |
| MAE     |   42,7941 |
| MSE     | 2900,1936 |
| RMSE    |   53,8534 |
| R²      |    0,4526 |

O valor de R² indica que o modelo apresentou capacidade de explicar aproximadamente 45% da variabilidade observada na variável-alvo no conjunto de teste.

Também foi construído um gráfico comparando os valores reais e previstos. A proximidade dos pontos em relação à linha de referência permite visualizar o grau de concordância entre as previsões e os valores observados.

## 6. Ridge e Lasso

Foram utilizados os métodos de regularização Ridge e Lasso com diferentes valores do parâmetro `alpha`: 0,001; 0,01; 0,1; 1; 10 e 100.

Para cada valor de `alpha`, foi calculado o erro quadrático médio no conjunto de teste.

Para Ridge, os valores de MSE foram:

| Alpha |       MSE |
| ----: | --------: |
| 0,001 | 2895,8212 |
|  0,01 | 2882,2902 |
|   0,1 | 2856,4869 |
|     1 | 3077,4159 |
|    10 | 4443,9526 |
|   100 | 5233,6637 |

Para Lasso:

| Alpha |       MSE |
| ----: | --------: |
| 0,001 | 2896,4098 |
|  0,01 | 2878,5594 |
|   0,1 | 2798,1935 |
|     1 | 3403,5757 |
|    10 | 5361,5335 |
|   100 | 5361,5335 |

Observa-se que, dentro dos valores de `alpha` testados, o menor MSE para Ridge ocorreu em `alpha = 0,1`, enquanto para Lasso o menor MSE ocorreu também em `alpha = 0,1`.

O aumento excessivo de `alpha` levou a um aumento do MSE nos dois métodos. Isso mostra que a intensidade da regularização influencia diretamente o desempenho dos modelos.

## 7. Regressão Polinomial

A Regressão Polinomial foi utilizada para investigar se relações não lineares poderiam melhorar a capacidade preditiva do modelo.

Foram avaliados graus de 1 a 6. Antes da geração das características polinomiais foi utilizado `StandardScaler`, seguido de `PolynomialFeatures` e `LinearRegression`.

Os resultados obtidos foram:

| Grau | MSE Treino | R² Treino | MSE Teste | R² Teste |
| ---: | ---------: | --------: | --------: | -------: |
|    1 |    2868,55 |    0,5279 |   2900,19 |   0,4526 |
|    2 |    2393,14 |    0,6062 |   3096,03 |   0,4156 |
|    3 |     745,62 |    0,8773 |  82446,05 | -14,5613 |
|    4 |        ≈ 0 |    1,0000 | 146484,25 | -26,6482 |
|    5 |        ≈ 0 |    1,0000 | 167972,17 | -30,7039 |
|    6 |        ≈ 0 |    1,0000 | 334595,03 | -62,1532 |

Os resultados mostram um comportamento claro de overfitting a partir do aumento do grau polinomial. Enquanto o desempenho no conjunto de treinamento melhora fortemente, chegando a R² igual a 1 nos graus 4, 5 e 6, o desempenho no conjunto de teste piora de maneira acentuada.

No grau 3, por exemplo, o R² de treinamento é 0,8773, enquanto o R² de teste é -14,5613. Nos graus superiores, a diferença aumenta ainda mais.

Esse comportamento indica que o modelo passa a se ajustar excessivamente aos dados de treinamento, perdendo capacidade de generalização para novos dados.

## 8. Decision Tree Regressor

Também foi utilizado um modelo de árvore de decisão para regressão.

Foram definidos os seguintes parâmetros:

`max_depth = 5`, `min_samples_split = 10`, `min_samples_leaf = 5` e `random_state = 42`.

Esses parâmetros limitam a complexidade da árvore e estabelecem uma quantidade mínima de observações necessária para determinadas divisões e folhas.

No conjunto de teste, a Decision Tree apresentou:

| Métrica | Resultado |
| ------- | --------: |
| MAE     |   43,4187 |
| MSE     | 2972,5166 |
| RMSE    |   54,5208 |
| R²      |    0,4390 |

O modelo apresentou desempenho próximo ao da Regressão Linear, porém com R² ligeiramente inferior no conjunto de teste. Também foi gerado um gráfico da árvore, permitindo visualizar parte da estrutura de decisões criada pelo modelo.

## 9. Random Forest Regressor

O modelo Random Forest foi implementado utilizando 100 árvores de decisão. Foram utilizados `max_depth = 5`, `min_samples_split = 10`, `min_samples_leaf = 5` e `random_state = 42`.

O modelo apresentou os seguintes resultados no conjunto de teste:

| Métrica | Resultado |
| ------- | --------: |
| MAE     |   43,8428 |
| MSE     | 2869,0728 |
| RMSE    |   53,5637 |
| R²      |    0,4585 |

O Random Forest apresentou MSE de 2869,0728 e RMSE de 53,5637 no conjunto de teste. Seu R² foi 0,4585.

Também foi calculada a importância das características no modelo. Os valores obtidos foram:

| Feature | Importância |
| ------- | ----------: |
| bmi     |      0,4395 |
| s5      |      0,2762 |
| bp      |      0,0754 |
| s6      |      0,0557 |
| s2      |      0,0367 |
| s3      |      0,0339 |
| age     |      0,0298 |
| s1      |      0,0289 |
| s4      |      0,0175 |
| sex     |      0,0063 |

A variável `bmi` apresentou a maior importância entre as características utilizadas pelo modelo, seguida por `s5`. A variável `sex` apresentou a menor importância entre as dez características consideradas pelo Random Forest.

Esses valores representam a importância das características para as decisões realizadas pelo modelo Random Forest e não devem ser interpretados isoladamente como relações causais.

## 10. Comparação dos modelos

A comparação dos modelos avaliados no conjunto de teste pode ser apresentada da seguinte maneira:

| Modelo           |     MAE |       MSE |    RMSE |     R² |
| ---------------- | ------: | --------: | ------: | -----: |
| Regressão Linear | 42,7941 | 2900,1936 | 53,8534 | 0,4526 |
| Decision Tree    | 43,4187 | 2972,5166 | 54,5208 | 0,4390 |
| Random Forest    | 43,8428 | 2869,0728 | 53,5637 | 0,4585 |

Os resultados mostram diferenças relativamente pequenas entre os três modelos no conjunto de teste. A Regressão Linear apresentou o menor MAE entre os três modelos, enquanto a Random Forest apresentou o menor MSE e RMSE e o maior R² entre esses modelos.

A Decision Tree apresentou os maiores valores de MSE e RMSE e o menor R² entre os três modelos avaliados.

É importante observar que a comparação deve considerar simultaneamente as diferentes métricas, pois cada uma avalia um aspecto diferente do erro de previsão.

## 11. Análise de overfitting

O principal indício de overfitting observado no projeto ocorreu na Regressão Polinomial.

Com o aumento do grau do polinômio, o desempenho no conjunto de treinamento melhorou progressivamente. Entretanto, o desempenho no conjunto de teste piorou drasticamente.

No grau 4, por exemplo, o R² de treinamento atingiu 1,0000, enquanto o R² de teste foi -26,6482. Esse comportamento demonstra uma grande diferença entre o ajuste aos dados conhecidos e a capacidade de previsão para dados não utilizados no treinamento.

A análise mostra, portanto, que aumentar a complexidade do modelo não necessariamente melhora sua capacidade de generalização.

## 12. Considerações sobre a validação

A validação cruzada foi aplicada à Regressão Linear utilizando seis folds. A média dos resultados apresentou MAE de 44,7702, MSE de 3047,5143, RMSE de 55,0825 e R² de 0,4814.

A validação cruzada permite avaliar o comportamento do modelo em diferentes subdivisões do conjunto de treinamento, fornecendo uma visão complementar à avaliação realizada no conjunto de teste.

No entanto, a implementação utilizada no projeto foi realizada manualmente e sem embaralhamento dos dados antes da divisão dos folds. Dessa forma, essa característica deve ser considerada como uma limitação metodológica da implementação.

## 13. Limitações

Uma das limitações do projeto é que nem todos os modelos foram avaliados utilizando exatamente o mesmo conjunto de métricas e procedimento de validação. Por exemplo, Ridge e Lasso foram comparados principalmente por meio do MSE para diferentes valores de `alpha`, enquanto os modelos Linear, Decision Tree e Random Forest foram avaliados utilizando MAE, MSE, RMSE e R².

Outra limitação é a utilização de uma única divisão entre treinamento e teste para a avaliação final. Embora a validação cruzada tenha sido utilizada para a Regressão Linear, ela não foi aplicada da mesma forma aos demais modelos.

A Regressão Polinomial também apresentou forte aumento da complexidade para graus elevados, produzindo diferenças muito grandes entre os resultados de treinamento e teste.

Além disso, algumas funções e bibliotecas importadas no código não são utilizadas diretamente na análise final, como modelos de classificação, PCA e alguns datasets adicionais. Esses elementos não interferem nos resultados obtidos, mas poderiam ser removidos para tornar o código mais organizado.

## 14. Conclusão

Neste projeto foi realizada uma análise de dados e aplicação de diferentes métodos de Machine Learning ao dataset Diabetes. Inicialmente foram exploradas as características do conjunto de dados por meio de estatísticas descritivas, gráficos de distribuição, gráficos de dispersão e matriz de correlação. Em seguida, os dados foram divididos em conjuntos de treinamento e teste e foram aplicados diferentes modelos de regressão. A Regressão Linear apresentou R² de 0,4526 no conjunto de teste, enquanto a Decision Tree apresentou R² de 0,4390 e a Random Forest apresentou R² de 0,4585.

A análise da Regressão Polinomial demonstrou que o aumento excessivo da complexidade pode produzir overfitting. Nos graus mais elevados, o modelo apresentou ajuste praticamente perfeito aos dados de treinamento, mas desempenho muito inferior no conjunto de teste.

Os modelos Ridge e Lasso também foram analisados com diferentes valores de regularização, sendo observado que valores excessivamente elevados de `alpha` aumentaram o erro quadrático médio no conjunto de teste.

Por fim, a análise das importâncias das características no Random Forest mostrou maior participação de `bmi` e `s5` nas decisões do modelo. De maneira geral, o projeto demonstrou a importância de combinar análise exploratória, treinamento, validação e métricas de desempenho para avaliar modelos de Machine Learning e compreender suas limitações.

# Perguntas para a conclusão

**1. Qual era o objetivo do projeto?**

O objetivo do projeto foi aplicar e comparar diferentes modelos de Machine Learning para prever a progressão da doença a partir de características clínicas e bioquímicas presentes no dataset Diabetes. Para isso, foram utilizados modelos de Regressão Linear, Ridge, Lasso, Regressão Polinomial, Decision Tree e Random Forest. Além da aplicação dos modelos, foram realizadas análises estatísticas e exploratórias dos dados, validação cruzada e avaliação do desempenho por meio de diferentes métricas de regressão.

**2. Qual dataset foi escolhido?**

Foi escolhido o Diabetes Dataset, disponibilizado pela biblioteca Scikit-learn. O conjunto de dados possui 442 observações, 10 variáveis preditoras e uma variável-alvo denominada `target`, relacionada à progressão da doença. As variáveis utilizadas como entrada foram `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5` e `s6`. O dataset foi escolhido por apresentar uma variável-alvo contínua, permitindo a aplicação de diferentes métodos de regressão e a comparação entre modelos com diferentes características.

**3. Quais cuidados foram necessários na preparação?**

Na preparação dos dados, inicialmente foram verificadas as dimensões do dataset, os tipos de dados, a presença de valores ausentes e a existência de registros duplicados. Também foram analisadas as estatísticas descritivas e as correlações entre as variáveis. Não foram identificados valores ausentes ou registros duplicados. Em seguida, a variável `target` foi separada das demais variáveis, formando os conjuntos `X` e `y`. Os dados foram então divididos em 80% para treinamento e 20% para teste, utilizando `random_state=42`, de modo que a divisão pudesse ser reproduzida. Na Regressão Polinomial, também foi utilizada a padronização das variáveis por meio do `StandardScaler` antes da criação das características polinomiais. Além disso, foi realizada uma validação cruzada com seis folds para a Regressão Linear.

**4. Por que o modelo foi escolhido?**

O projeto não utilizou apenas um modelo, mas diferentes métodos de regressão com o objetivo de comparar suas características e seus desempenhos. A Regressão Linear foi utilizada como uma abordagem de referência por representar relações lineares entre as variáveis. Os modelos Ridge e Lasso foram utilizados para analisar o efeito da regularização. A Regressão Polinomial foi aplicada para investigar a possibilidade de relações não lineares entre as variáveis. Já a Decision Tree permite realizar previsões por meio de regras de decisão, enquanto a Random Forest utiliza um conjunto de árvores de decisão para realizar as previsões. Dessa forma, a utilização de diferentes modelos permitiu comparar abordagens distintas para o mesmo problema de previsão.

**5. Por que essas métricas foram utilizadas?**

Foram utilizadas as métricas MAE, MSE, RMSE e R² porque cada uma fornece uma perspectiva diferente sobre o desempenho dos modelos. O MAE representa o erro absoluto médio entre os valores reais e os valores previstos, sendo uma métrica de interpretação relativamente simples. O MSE calcula o erro quadrático médio e atribui maior peso aos erros de maior magnitude. O RMSE corresponde à raiz quadrada do MSE e apresenta a vantagem de estar na mesma escala da variável-alvo. Já o R² permite avaliar a proporção da variabilidade da variável-alvo que é explicada pelo modelo. A utilização conjunta dessas métricas permite realizar uma avaliação mais completa dos modelos e de seus erros de previsão.

**6. O resultado foi satisfatório? Justifique com números.**

Os resultados obtidos indicam uma capacidade preditiva moderada dos modelos avaliados. No conjunto de teste, a Regressão Linear apresentou MAE de 42,7941, MSE de 2900,1936, RMSE de 53,8534 e R² de 0,4526. A Decision Tree apresentou MAE de 43,4187, MSE de 2972,5166, RMSE de 54,5208 e R² de 0,4390. Já a Random Forest apresentou MAE de 43,8428, MSE de 2869,0728, RMSE de 53,5637 e R² de 0,4585. Portanto, os modelos conseguiram explicar aproximadamente 44% a 46% da variabilidade observada no conjunto de teste. A Random Forest apresentou o maior R² entre esses três modelos, com 0,4585, além do menor MSE e RMSE, enquanto a Regressão Linear apresentou o menor MAE. Dessa forma, os resultados podem ser considerados razoáveis para o projeto, embora ainda exista uma parcela significativa da variabilidade que não é explicada pelos modelos.

**7. Existem sinais de overfitting ou underfitting?**

Foram observados sinais claros de overfitting na Regressão Polinomial, principalmente nos graus mais elevados. No grau 4, por exemplo, o modelo apresentou R² de treinamento igual a 1,0000, enquanto o R² no conjunto de teste foi de -26,6482. No grau 6, o R² de treinamento também foi 1,0000, mas o R² de teste caiu para -62,1532. Esse comportamento indica que o modelo se ajustou excessivamente aos dados de treinamento e perdeu capacidade de generalização para dados não utilizados durante o treinamento. Em relação à Decision Tree e à Random Forest, o código não apresenta os resultados dessas métricas no conjunto de treinamento, portanto não é possível afirmar com a mesma segurança a existência de overfitting ou underfitting nesses modelos apenas com os resultados obtidos. A validação cruzada da Regressão Linear apresentou R² médio de 0,4814, enquanto o R² no teste final foi 0,4526, indicando resultados relativamente próximos entre essas avaliações.

**8. O que poderia ser melhorado?**

O projeto poderia ser melhorado principalmente por meio de uma avaliação mais padronizada dos modelos. Uma possibilidade seria aplicar validação cruzada a todos os modelos, permitindo comparar seus desempenhos utilizando exatamente o mesmo procedimento. Também poderia ser realizada uma busca sistemática dos hiperparâmetros, utilizando métodos como Grid Search ou Randomized Search, em vez de testar apenas alguns valores definidos manualmente. Outra melhoria seria avaliar todos os modelos utilizando simultaneamente MAE, MSE, RMSE e R², facilitando a comparação. A análise poderia ainda ser complementada com gráficos de resíduos, permitindo investigar melhor os erros das previsões. No caso da Regressão Polinomial, seria importante controlar a complexidade do modelo para evitar o forte overfitting observado nos graus mais elevados. Por fim, o código poderia ser organizado de maneira mais clara, removendo bibliotecas e modelos que não são utilizados na análise final e estruturando melhor as etapas de preparação dos dados, treinamento, avaliação e comparação dos modelos.
