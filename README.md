# MarketLab
# Advanced Market Analysis

Este projeto implementa uma análise avançada de mercado para criptomoedas, utilizando dados históricos da Binance para classificar estados de mercado e gerar sinais de trading.

## Funcionalidades

- Recuperação de dados históricos da API da Binance
- Classificação avançada de estados de mercado
- Geração de sinais de trading baseados em análise probabilística
- Visualização de sinais de compra e venda

## Requisitos

- Python 3.7+
- Bibliotecas: numpy, pandas, requests, scipy, matplotlib

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/JimSP/MarketLab
cd advanced-market-analysis
```

2. Instale as dependências:

```bash
pip install numpy pandas requests scipy matplotlib
```

## Uso

Para executar a análise, use o seguinte código:

```python
from advanced_market_analysis import AdvancedMarketAnalysis

analysis = AdvancedMarketAnalysis()
results = analysis.run_analysis()

print("Resumo de Sinais:")
print(results['signal_summary'])
print("\nEstatísticas de Confiança:")
print(results['confidence_stats'])
```

## Configuração

Você pode ajustar os seguintes parâmetros na inicialização da classe `AdvancedMarketAnalysis`:

- `symbol`: Par de trading (padrão: 'BTCUSDT')
- `interval`: Intervalo de tempo para os candles (padrão: '1h')
- `limit`: Número de candles a serem recuperados (padrão: 1000)

## Contribuição

Contribuições são bem-vindas! Por favor, siga estas etapas:

1. Fork o projeto
2. Crie sua branch de feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## Contato

Link do Projeto: [https://github.com/JimSP/MarketLab](https://github.com/JimSP/MarketLab)
