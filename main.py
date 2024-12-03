import numpy as np
import pandas as pd
import requests
from scipy import stats
import matplotlib.pyplot as plt
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AdvancedMarketAnalysis:
    def _init_(self, symbol='BTCUSDT', interval='1h', limit=1000):
        """
        Inicializa a classe com os parâmetros fornecidos e busca os dados históricos da Binance.
        """
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.df = self.get_binance_klines()
        if self.df.empty:
            logging.error("DataFrame vazio. Verifique a conexão com a API ou os parâmetros fornecidos.")

    def get_binance_klines(self):
        """
        Recupera dados históricos de candles da API da Binance e os estrutura em um DataFrame.
        """
        url = "https://api.binance.us/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Levanta HTTPError para respostas ruins
            data = response.json()
            logging.info("Dados de candles recuperados com sucesso.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao buscar dados: {e}")
            return pd.DataFrame()  # Retorna DataFrame vazio ou tratar conforme necessário

        # Definir os nomes das colunas conforme a resposta da API
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ]

        df = pd.DataFrame(data, columns=columns)

        # Converter as colunas relevantes para float
        try:
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            logging.info("Dados convertidos para float com sucesso.")
        except KeyError as e:
            logging.error(f"Erro na conversão de dados: {e}")
            return pd.DataFrame()

        return df

    def classify_volume(self, volumes):
        """
        Classifica o volume atual em categorias hierárquicas.
        """
        q1, q3 = np.percentile(volumes, [25, 75])
        iqr = q3 - q1
        current_volume = volumes.iloc[-1]

        if current_volume >= (q3 + 1.5 * iqr):
            return 'Volume Extremamente Alto'
        elif current_volume >= q3:
            return 'Alto Volume'
        elif current_volume >= q1:
            return 'Volume Médio'
        elif current_volume >= (q1 - 1.5 * iqr):
            return 'Baixo Volume'
        else:
            return 'Muito Baixo Volume'

    def classify_volatility(self, price_change):
        """
        Classifica a volatilidade com base no desvio padrão das mudanças de preço.
        """
        std_dev = price_change.std()
        quantiles = price_change.quantile([0.2, 0.4, 0.6, 0.8])

        if std_dev >= quantiles[0.8]:
            return 'Volatilidade Extrema'
        elif std_dev >= quantiles[0.6]:
            return 'Alta Volatilidade'
        elif std_dev >= quantiles[0.4]:
            return 'Volatilidade Média'
        elif std_dev >= quantiles[0.2]:
            return 'Baixa Volatilidade'
        else:
            return 'Baixíssima Volatilidade'

    def classify_trend(self, closes):
        """
        Classifica a tendência com base na inclinação da regressão linear das fechamentos.
        """
        x = np.arange(len(closes))
        slope, _, _, _, _ = stats.linregress(x, closes)

        std_dev = closes.std()

        if slope > 2 * std_dev:
            return 'Forte Tendência de Alta'
        elif slope > std_dev:
            return 'Tendência de Alta Moderada'
        elif abs(slope) <= 0.5 * std_dev:
            return 'Neutro'
        elif slope < -std_dev:
            return 'Tendência de Baixa Moderada'
        else:
            return 'Forte Tendência de Baixa'

    def classify_momentum(self, price_change):
        """
        Classifica o momentum com base na média das mudanças de preço.
        """
        momentum = price_change.mean()
        std_dev = price_change.std()

        if momentum > 2 * std_dev:
            return 'Momentum Comprador Forte'
        elif momentum > 0:
            return 'Momentum Comprador Fraco'
        elif abs(momentum) < std_dev:
            return 'Momentum Neutro'
        elif momentum < -2 * std_dev:
            return 'Momentum Vendedor Forte'
        else:
            return 'Momentum Vendedor Fraco'

    def classify_market_structure(self, highs, lows):
        """
        Classifica a estrutura do mercado com base nos máximos e mínimos atuais.
        """
        max_high = highs.max()
        min_low = lows.min()
        current_high = highs.iloc[-1]
        current_low = lows.iloc[-1]

        if current_high == max_high and current_low > min_low:
            return 'Breakout Inicial'
        elif current_high < max_high and current_low > min_low:
            return 'Consolidação'
        elif current_high < max_high and current_low < min_low:
            return 'Pré-Reversão'
        else:
            return 'Continuação de Tendência'  # Padrão

    def get_composite_market_state(self, market_state):
        """
        Determina o estado composto do mercado com base nos estados individuais.
        """
        if (market_state['trend_state'] == 'Neutro' and
            market_state['volatility_state'] in ['Volatilidade Extrema', 'Alta Volatilidade'] and
            market_state['momentum_state'] == 'Momentum Comprador Fraco'):
            return 'Consolidação Volátil'

        if (market_state['volatility_state'] in ['Volatilidade Extrema', 'Alta Volatilidade'] and
            market_state['momentum_state'] in ['Momentum Comprador Fraco', 'Momentum Vendedor Fraco']):
            return 'Tendência Instável'

        # Adicione mais condições compostas conforme necessário

        return 'Estado Composto Padrão'  # Padrão

    def advanced_market_state_classifier(self, window):
        """
        Classifica o estado do mercado com base em uma janela de dados.
        """
        closes = window['close']
        opens = window['open']
        highs = window['high']
        lows = window['low']
        volumes = window['volume']

        price_change = closes.pct_change().dropna()
        volume_change = volumes.pct_change().dropna()

        # Classificações individuais
        volume_state = self.classify_volume(volumes)
        volatility_state = self.classify_volatility(price_change)
        trend_state = self.classify_trend(closes)
        momentum_state = self.classify_momentum(price_change)
        structure_state = self.classify_market_structure(highs, lows)

        # Estado composto
        composite_state = self.get_composite_market_state({
            'volume_state': volume_state,
            'volatility_state': volatility_state,
            'trend_state': trend_state,
            'momentum_state': momentum_state,
            'structure_state': structure_state
        })

        market_state = {
            'volume_state': volume_state,
            'volatility_state': volatility_state,
            'trend_state': trend_state,
            'momentum_state': momentum_state,
            'structure_state': structure_state,
            'composite_state': composite_state
        }

        return market_state

    def analyze_market_states(self, window_size=50):
        """
        Analisa os estados de mercado ao longo de todo o DataFrame utilizando uma janela deslizante.
        """
        if self.df.empty:
            logging.error("DataFrame vazio. Não é possível analisar os estados do mercado.")
            return []

        market_states = []

        for i in range(len(self.df) - window_size + 1):
            window = self.df.iloc[i:i + window_size]
            states = self.advanced_market_state_classifier(window)
            market_states.append(states)

            if i % 100 == 0:
                logging.info(f"Analisadas {i} janelas de mercado.")

        logging.info("Análise dos estados de mercado concluída.")
        return market_states

    def generate_trading_signals(self, market_states, lookback=3):
        """
        Gera sinais de trading com base nos estados de mercado analisados.
        """
        signals = []

        for i in range(len(market_states) - lookback + 1):
            recent_states = market_states[i:i + lookback]

            # Condições probabilísticas de sinal
            volume_prob = sum(state['volume_state'] == 'Volume Médio' for state in recent_states) / lookback
            volatility_prob = sum(state['volatility_state'] in ['Volatilidade Extrema', 'Alta Volatilidade'] for state in recent_states) / lookback

            # Sinais de compra
            if volume_prob > 0.7 and volatility_prob > 0.6:
                signals.append({
                    'signal': 'BUY_POTENTIAL',
                    'confidence': volume_prob * volatility_prob,
                    'details': recent_states[-1]
                })
            # Sinais de venda
            elif all(state['volume_state'] == 'Baixo Volume' for state in recent_states):
                sell_confidence = sum(state['momentum_state'] == 'Momentum Vendedor Fraco' for state in recent_states) / lookback
                signals.append({
                    'signal': 'SELL_POTENTIAL',
                    'confidence': sell_confidence,
                    'details': recent_states[-1]
                })
            else:
                signals.append({
                    'signal': 'HOLD',
                    'confidence': 0,
                    'details': recent_states[-1]
                })

        logging.info("Geração de sinais de trading concluída.")
        return signals

    def visualize_signals(self, signals, window_size, lookback):
        """
        Visualiza os sinais de compra e venda sobrepostos aos preços de fechamento.
        """
        if self.df.empty:
            logging.error("DataFrame vazio. Não é possível visualizar os sinais.")
            return

        buy_signals = []
        sell_signals = []
        buy_indices = []
        sell_indices = []

        # O índice inicial dos sinais após considerar a janela e o lookback
        start_index = window_size + lookback - 1

        for i, signal in enumerate(signals):
            actual_index = i + start_index
            # Verificar se o índice está dentro dos limites do DataFrame
            if actual_index >= len(self.df):
                logging.warning(f"Índice {actual_index} está fora dos limites do DataFrame. Ignorando.")
                continue
            if signal['signal'] == 'BUY_POTENTIAL':
                buy_signals.append(self.df['close'].iloc[actual_index])
                buy_indices.append(actual_index)
            elif signal['signal'] == 'SELL_POTENTIAL':
                sell_signals.append(self.df['close'].iloc[actual_index])
                sell_indices.append(actual_index)

        plt.figure(figsize=(15, 6))
        plt.plot(self.df['close'].values, label='Preço de Fechamento')
        plt.scatter(buy_indices, buy_signals, color='green', label='Sinal de Compra', marker='^', alpha=1)
        plt.scatter(sell_indices, sell_signals, color='red', label='Sinal de Venda', marker='v', alpha=1)
        plt.title('Sinais de Trading')
        plt.xlabel('Período')
        plt.ylabel('Preço (USDT)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_analysis(self, window_size=50, lookback=3):
        """
        Executa toda a análise de mercado, desde a classificação dos estados até a geração e visualização dos sinais.
        """
        logging.info("Iniciando análise de mercado...")
        # Análise dos estados de mercado
        market_states = self.analyze_market_states(window_size)
        # Geração de sinais de trading
        trading_signals = self.generate_trading_signals(market_states, lookback)

        # Sumário de sinais
        signal_summary = {
            'total_signals': len(trading_signals),
            'buy_signals': sum(1 for signal in trading_signals if signal['signal'] == 'BUY_POTENTIAL'),
            'sell_signals': sum(1 for signal in trading_signals if signal['signal'] == 'SELL_POTENTIAL'),
            'hold_signals': sum(1 for signal in trading_signals if signal['signal'] == 'HOLD')
        }

        # Estatísticas de confiança
        buy_confidences = [s['confidence'] for s in trading_signals if s['signal'] == 'BUY_POTENTIAL']
        sell_confidences = [s['confidence'] for s in trading_signals if s['signal'] == 'SELL_POTENTIAL']
        confidence_stats = {
            'avg_buy_confidence': np.mean(buy_confidences) if buy_confidences else 0,
            'avg_sell_confidence': np.mean(sell_confidences) if sell_confidences else 0
        }

        # Visualização dos sinais
        self.visualize_signals(trading_signals, window_size, lookback)

        logging.info("Análise de mercado concluída.")
        return {
            'market_states': market_states,
            'trading_signals': trading_signals,
            'signal_summary': signal_summary,
            'confidence_stats': confidence_stats
        }


if _name_ == "_main_":
    # Executar análise
    analysis = AdvancedMarketAnalysis()
    results = analysis.run_analysis()

    # Imprimir resultados detalhados
    print("Resumo de Sinais:")
    print(results['signal_summary'])
    print("\nEstatísticas de Confiança:")
    print(results['confidence_stats'])