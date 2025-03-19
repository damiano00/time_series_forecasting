# Title
Time Series Analysis and Forecasting Applied to Automated Financial Trading

## Abstract

Recent advances in deep reinforcement learning (DRL) have led to improved stock market prediction and trading strategies. One notable state-of-the-art method is the cascaded LSTM with Proximal Policy Optimization (CLSTM-PPO), which combines sequence modeling and policy optimization to achieve superior trading performance on global stock indices. However, existing DRL-based models largely rely on price and technical indicators alone, overlooking the informational value of textual sentiment data. This thesis proposes the usage of a recent dataset named FNSPID, which contains both stock price data in the form of time series and sentiment features extracted from financial news articles, to enhance the performance of the CLSTM-PPO model. The accuracy of the sentiment-informed model and the non-sentiment-informed model are compared against the S.o.A. deep learning models on the FNSPID dataset. Our experiments show that adding sentiment features notably improves prediction accuracy and trading returns, validating the benefit of incorporating qualitative information into RL-based forecasting. However, we observed that these improvements are constrained by data coverage limitations: the current dataset for different stocks does not provide a sufficiently long overlapping timeframe, which limits the model's training horizon. Thus, while sentiment data clearly augments predictive performance, the dataset must be extended to fully capitalize on it, also considering the remarkable potential of the FNSPID's extensibility. Future work should expand FNSPID to a broader, aligned time period across multiple stocks, enabling more robust model training and more accurate, generalizable stock market predictions.

## License

[Attribution-NonCommercial 4.0 International, Version 4.0, January 2004](https://creativecommons.org/licenses/by-nc/4.0/)


## Acknowledgements

 - [Dong, Zihan, Xinyu Fan, and Zhiyuan Peng. "Fnspid: A comprehensive financial news dataset in time series." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.](https://dl.acm.org/doi/abs/10.1145/3637528.3671629)

 - [Zou, Jie, et al. "A novel deep reinforcement learning based automated stock trading system using cascaded lstm networks." Expert Systems with Applications 242 (2024): 122801.](https://www.sciencedirect.com/science/article/pii/S0957417423033031)
## Author

- [Damiano Pasquini (@damiano00)](https://www.github.com/damiano00)

## Supervisors

- [Michela Quadrini](https://scholar.google.com/citations?user=7iGg5wQAAAAJ&hl=en&oi=ao) [@UNICAM](https://www.unicam.it/en/home)
- [María Óskarsdóttir](https://scholar.google.com/citations?user=-R5x1_QAAAAJ&hl=en&oi=ao) [@RU](https://www.ru.is/en)
