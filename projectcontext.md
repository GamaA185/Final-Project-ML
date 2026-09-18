### Project Context

Este projeto tem como objetivo aplicar técnicas de Machine Learning para analisar e prever a progressão de uma doença a partir de diferentes características clínicas e bioquímicas. Para isso, foi utilizado o Diabetes Dataset, disponibilizado pela biblioteca Scikit-learn, composto por 442 observações, 10 variáveis preditoras e uma variável-alvo contínua relacionada à progressão da doença.

Inicialmente, foi realizada uma análise exploratória dos dados, incluindo a verificação das dimensões, tipos das variáveis, valores ausentes, registros duplicados, estatísticas descritivas e correlações entre as variáveis. Em seguida, os dados foram divididos em conjuntos de treinamento e teste, permitindo avaliar a capacidade dos modelos de generalizar para dados não utilizados durante o treinamento.

Foram implementados e comparados diferentes modelos de regressão, incluindo Regressão Linear, Ridge, Lasso, Regressão Polinomial, Decision Tree e Random Forest. A utilização de diferentes abordagens teve como finalidade verificar como modelos lineares, modelos com regularização e métodos capazes de representar relações não lineares se comportam diante do mesmo conjunto de dados.

O desempenho dos modelos foi avaliado utilizando as métricas MAE, MSE, RMSE e R². Além disso, foi realizada validação cruzada para a Regressão Linear e foram analisados diferentes graus na Regressão Polinomial, permitindo investigar o efeito da complexidade do modelo sobre sua capacidade de generalização. Dessa forma, o projeto busca não apenas realizar previsões, mas também compreender as características, limitações e comportamentos dos diferentes modelos de Machine Learning aplicados ao problema.
