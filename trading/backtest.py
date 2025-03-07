import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class Backtest:
    """
    Clase para ejecutar backtests de estrategias de trading.
    """
    
    def __init__(self, df_signals: pd.DataFrame):
        """
        Inicializa el objeto con los datos del mercado y señales.
        
        Args:
            df_signals: DataFrame con datos de mercado e indicadores y señales
        """
        self.df = df_signals.copy()
        
        # Estandarizar nombres de columnas
        if 'Close' in self.df.columns and 'close' not in self.df.columns:
            self.df['close'] = self.df['Close']
            self.price_col = 'close'
        elif 'close' in self.df.columns:
            self.price_col = 'close'
        else:
            # Si no encontramos ninguna, intentamos usar otra columna
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                self.price_col = numeric_cols[0]
                print(f"ADVERTENCIA: No se encontró columna 'close'/'Close'. Usando '{self.price_col}' para precios.")
            else:
                raise ValueError("No se encontraron columnas numéricas para usar como precio.")
        
        # Verificar que existe la columna de señales
        if 'signal' not in self.df.columns:
            raise ValueError("La columna 'signal' es necesaria para el backtest")
    
    def run(self, 
           initial_capital: float = 10000.0, 
           position_size: float = 1.0,
           commission: float = 0.001) -> Dict[str, Any]:
        """
        Ejecuta el backtest y calcula métricas de rendimiento.
        
        Args:
            initial_capital: Capital inicial para el backtest
            position_size: Tamaño de la posición (0.0-1.0 como porcentaje del capital)
            commission: Comisión por operación (como porcentaje)
            
        Returns:
            Diccionario con los resultados del backtest
        """
        df_backtest = self.df.copy()
        
        # Inicializar columnas para el backtest
        df_backtest['capital'] = initial_capital
        df_backtest['shares'] = 0
        df_backtest['cash'] = initial_capital
        df_backtest['trade_price'] = 0.0
        df_backtest['trade_value'] = 0.0
        df_backtest['commission_paid'] = 0.0
        df_backtest['return'] = 0.0
        
        # Iterar a través de los datos para simular operaciones
        for i in range(1, len(df_backtest)):
            # Copiar valores iniciales del día anterior
            df_backtest.loc[df_backtest.index[i], 'capital'] = df_backtest.loc[df_backtest.index[i-1], 'capital']
            df_backtest.loc[df_backtest.index[i], 'shares'] = df_backtest.loc[df_backtest.index[i-1], 'shares']
            df_backtest.loc[df_backtest.index[i], 'cash'] = df_backtest.loc[df_backtest.index[i-1], 'cash']
            
            # Precio actual
            current_price = df_backtest.loc[df_backtest.index[i], self.price_col]
            
            # Verificar si hay una señal de trading
            signal = df_backtest.loc[df_backtest.index[i], 'signal']
            
            if signal != 0:  # Si hay una señal de compra o venta
                available_capital = df_backtest.loc[df_backtest.index[i], 'capital']
                
                if signal > 0:  # Señal de compra
                    # Si no tenemos posición, comprar
                    if df_backtest.loc[df_backtest.index[i], 'shares'] == 0:
                        # Calcular cantidad a invertir
                        invest_amount = available_capital * position_size
                        # Calcular número de acciones a comprar
                        num_shares = (invest_amount / current_price)
                        # Calcular comisión
                        commission_amount = invest_amount * commission
                        # Actualizar registros
                        df_backtest.loc[df_backtest.index[i], 'shares'] = num_shares
                        df_backtest.loc[df_backtest.index[i], 'cash'] -= (invest_amount + commission_amount)
                        df_backtest.loc[df_backtest.index[i], 'trade_price'] = current_price
                        df_backtest.loc[df_backtest.index[i], 'trade_value'] = invest_amount
                        df_backtest.loc[df_backtest.index[i], 'commission_paid'] = commission_amount
                
                elif signal < 0:  # Señal de venta
                    # Si tenemos posición, vender
                    if df_backtest.loc[df_backtest.index[i], 'shares'] > 0:
                        # Calcular valor de venta
                        num_shares = df_backtest.loc[df_backtest.index[i], 'shares']
                        sell_amount = num_shares * current_price
                        # Calcular comisión
                        commission_amount = sell_amount * commission
                        # Actualizar registros
                        df_backtest.loc[df_backtest.index[i], 'shares'] = 0
                        df_backtest.loc[df_backtest.index[i], 'cash'] += (sell_amount - commission_amount)
                        df_backtest.loc[df_backtest.index[i], 'trade_price'] = current_price
                        df_backtest.loc[df_backtest.index[i], 'trade_value'] = -sell_amount
                        df_backtest.loc[df_backtest.index[i], 'commission_paid'] = commission_amount
            
            # Actualizar capital (efectivo + valor de las acciones)
            shares_value = df_backtest.loc[df_backtest.index[i], 'shares'] * current_price
            df_backtest.loc[df_backtest.index[i], 'capital'] = df_backtest.loc[df_backtest.index[i], 'cash'] + shares_value
            
            # Calcular retorno diario
            prev_capital = df_backtest.loc[df_backtest.index[i-1], 'capital']
            curr_capital = df_backtest.loc[df_backtest.index[i], 'capital']
            df_backtest.loc[df_backtest.index[i], 'return'] = (curr_capital / prev_capital) - 1
        
        # Calcular métricas de rendimiento
        returns = df_backtest['return'].dropna()
        
        # Retorno total
        total_return = (df_backtest['capital'].iloc[-1] / initial_capital) - 1
        
        # Retorno anualizado (asumiendo 252 días de trading al año)
        trading_days = len(returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatilidad
        daily_std = returns.std()
        annualized_std = daily_std * np.sqrt(252)
        
        # Ratio de Sharpe (asumiendo tasa libre de riesgo de 0 para simplificar)
        sharpe_ratio = annualized_return / annualized_std if annualized_std > 0 else 0
        
        # Máximo drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Recopilar resultados
        results = {
            'initial_capital': initial_capital,
            'final_capital': df_backtest['capital'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trading_days': trading_days,
            'total_trades': (df_backtest['trade_value'] != 0).sum(),
            'total_commission': df_backtest['commission_paid'].sum(),
            'backtest_data': df_backtest
        }
        
        return results
    
    def plot_equity_curve(self, results: Optional[Dict[str, Any]] = None):
        """
        Genera un gráfico de la curva de equidad del backtest.
        Si se usa desde un notebook, mostrará el gráfico. Si no, devolverá la figura.
        
        Args:
            results: Resultados del backtest (si es None, debe haberse ejecutado run() antes)
            
        Returns:
            Figura de matplotlib (si no se está ejecutando en un notebook)
        """
        import matplotlib.pyplot as plt
        
        if results is None:
            # Verificar si tenemos datos de backtest
            if 'capital' not in self.df.columns:
                raise ValueError("No se han encontrado datos de backtest. Ejecute run() primero.")
            df_plot = self.df
        else:
            df_plot = results.get('backtest_data')
            if df_plot is None:
                raise ValueError("Los resultados no contienen datos de backtest.")
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot.index, df_plot['capital'], label='Capital')
        
        # Añadir líneas para operaciones
        buy_signals = df_plot[df_plot['signal'] > 0]
        sell_signals = df_plot[df_plot['signal'] < 0]
        
        plt.scatter(buy_signals.index, buy_signals['capital'], 
                   color='green', marker='^', alpha=0.7, s=100, label='Compra')
        plt.scatter(sell_signals.index, sell_signals['capital'], 
                   color='red', marker='v', alpha=0.7, s=100, label='Venta')
        
        plt.title('Curva de Equidad')
        plt.xlabel('Fecha')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Si estamos en un notebook, mostrar el gráfico
        # De lo contrario, devolver la figura
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
                return None
        except:
            pass
        
        return plt.gcf()