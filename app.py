import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pytz
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext
import threading
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import ta  # Biblioteca alternativa para indicadores técnicos

# Configurações para logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações da API da IQ Option
api = IQ_Option("fag.almeida2@gmail.com", "*Flavia01071999")  # Substitua por suas credenciais
check, reason = api.connect()
if not check:
    logger.error(f"Falha ao conectar: {reason}")
else:
    logger.info("Conexão bem-sucedida com a IQ Option")

# Moedas a serem analisadas
moedas = ['EURUSD', 'GBPUSD']

# Contadores para desempenho
tentativas = 0
erros = 0

# Função para obter dados de candles
def obter_dados_candles(moeda, intervalo='1m'):
    try:
        tempo_atual = time.time()
        response = api.get_candles(moeda, 60, 1000, tempo_atual)
        if isinstance(response, list) and len(response) > 0:
            df = pd.DataFrame(response)
            df['datetime'] = pd.to_datetime(df['from'], unit='s')
            df.set_index('datetime', inplace=True)
            # Renomear colunas para os nomes esperados
            df.rename(columns={'max': 'high', 'min': 'low'}, inplace=True)
            # Inspecionar as colunas disponíveis
            logger.info(f"Colunas disponíveis para {moeda}: {df.columns.tolist()}")
            return df
        else:
            logger.error(f"Formato inesperado da resposta para {moeda}: {response}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erro ao obter dados para {moeda}: {e}")
        return pd.DataFrame()

# Função para calcular indicadores técnicos usando ta
def calcular_indicadores(df):
    # Verifica se todas as colunas necessárias estão presentes
    required_columns = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        logger.error("Dados insuficientes para calcular indicadores.")
        return pd.DataFrame()
    
    df['MA5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['EMA10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    df.dropna(inplace=True)
    return df

# Função para gerar sinais futuros com Random Forest
def gerar_sinais_futuros(df):
    df = calcular_indicadores(df)
    if df.empty:
        return []

    df['target'] = df['close'].shift(-1) > df['close']
    df.dropna(inplace=True)
    
    features = ['MA5', 'EMA10', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D']
    X = df[features]
    y = df['target'].astype(int)
    
    # Balanceamento dos dados
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Normalização dos dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    grid_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42, error_score='raise')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Acurácia do modelo: {accuracy:.4f}")
    
    # Previsão para o próximo candle
    future_data = scaler.transform(X[-1].reshape(1, -1))
    future_pred = best_model.predict(future_data)[0]
    
    sinal = 'Compra' if future_pred else 'Venda'
    future_time = time.time() + 60
    
    return [{
        'Sinal': sinal,
        'datetime': datetime.fromtimestamp(future_time).strftime('%Y-%m-%d %H:%M:%S')
    }]

# Função para coletar sinais em tempo real
def coletar_sinais():
    global tentativas, erros
    sinais = []
    tz = pytz.timezone('America/Sao_Paulo')
    now = datetime.now(tz)

    for moeda in moedas:
        df = obter_dados_candles(moeda)
        if not df.empty:
            sinais_futuros = gerar_sinais_futuros(df)
            for sinal in sinais_futuros:
                tentativas += 1
                sinais.append({
                    'Moeda': moeda,
                    'Sinal': sinal['Sinal'],
                    'datetime': sinal['datetime']
                })
    
    logger.info(f"Sinais coletados: {sinais}")
    return sinais

def atualizar_sinais(text_widget, moeda, tipo_opcao):
    sinais = coletar_sinais()
    
    text_widget.delete(1.0, tk.END)
    
    header = f"Moeda | Sinal | Data e Hora | Tipo: {tipo_opcao}\n" + "-"*50
    text_widget.insert(tk.END, header + "\n")
    
    if sinais:
        for sinal in sinais:
            linha = f"{sinal['Moeda']} | {sinal['Sinal']} | {sinal['datetime']}\n"
            text_widget.insert(tk.END, linha)
    else:
        text_widget.insert(tk.END, "Nenhum sinal disponível.")

def exibir_janela():
    root = tk.Tk()
    root.title("Sinais de Trading")
    
    # Widgets para seleção
    tk.Label(root, text="Escolha a moeda:").pack(pady=5)
    moeda_var = tk.StringVar(value=moedas[0])
    tk.OptionMenu(root, moeda_var, *moedas).pack(pady=5)

    tk.Label(root, text="Escolha o tipo de opção:").pack(pady=5)
    tipo_var = tk.StringVar(value='binária')
    tk.OptionMenu(root, tipo_var, 'binária', 'digital').pack(pady=5)
    
    text_widget = scrolledtext.ScrolledText(root, width=80, height=20)
    text_widget.pack(padx=10, pady=10)
    
    atualizar_button = tk.Button(root, text="Atualizar Sinais", command=lambda: threading.Thread(target=atualizar_sinais, args=(text_widget, moeda_var.get(), tipo_var.get())).start())
    atualizar_button.pack(pady=5)
    
    atualizar_sinais(text_widget, moeda_var.get(), tipo_var.get())
    
    root.mainloop()

if __name__ == '__main__':
    exibir_janela()
