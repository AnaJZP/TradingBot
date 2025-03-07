import pandas as pd
import numpy as np

class CrossoverSignals:
    """
    Clase para generar señales de trading basadas en cruces de medias móviles.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el objeto con los datos del mercado incluyendo indicadores.
        
        Args:
            df: DataFrame con datos de mercado e indicadores técnicos
        """
        self.df = df.copy()
        
        # Estandarizar nombres de columnas
        if 'Close' in self.df.columns and 'close' not in self.df.columns:
            self.df['close'] = self.df['Close']
            self.price_col = 'close'
        elif 'close' in self.df.columns:
            self.price_col = 'close'
        else:
            # Si no encontramos ninguna, usamos la primera columna disponible
            # y advertimos sobre ello
            self.price_col = self.df.columns[0]
            print(f"ADVERTENCIA: No se encontró columna 'close'/'Close'. Usando '{self.price_col}' para precios.")
    
    def sma_ema_crossover(self, sma_period: int, ema_period: int) -> pd.DataFrame:
        """
        Genera señales basadas en el cruce de SMA y EMA.
        
        Estrategia:
        - Compra (1): Cuando EMA cruza por encima de SMA
        - Venta (-1): Cuando EMA cruza por debajo de SMA
        - Mantener (0): En cualquier otro caso
        
        Args:
            sma_period: Período de la SMA
            ema_period: Período de la EMA
            
        Returns:
            DataFrame con señales añadidas
        """
        df_signals = self.df.copy()
        
        # Asegurar que las columnas existen
        sma_col = f'SMA_{sma_period}'
        ema_col = f'EMA_{ema_period}'
        
        if sma_col not in df_signals.columns or ema_col not in df_signals.columns:
            raise ValueError(f"Columnas {sma_col} o {ema_col} no encontradas en el DataFrame")
        
        # Crear columna para señales
        df_signals['signal'] = 0
        
        # Generar señales de compra (1) cuando EMA cruza por encima de SMA
        df_signals.loc[(df_signals[ema_col] > df_signals[sma_col]) & 
                      (df_signals[ema_col].shift(1) <= df_signals[sma_col].shift(1)), 'signal'] = 1
        
        # Generar señales de venta (-1) cuando EMA cruza por debajo de SMA
        df_signals.loc[(df_signals[ema_col] < df_signals[sma_col]) & 
                      (df_signals[ema_col].shift(1) >= df_signals[sma_col].shift(1)), 'signal'] = -1
        
        # Añadir columna de posición acumulada (para backtesting)
        df_signals['position'] = df_signals['signal'].cumsum()
        
        return df_signals
    
    def double_sma_crossover(self, short_period: int, long_period: int) -> pd.DataFrame:
        """
        Genera señales basadas en el cruce de dos SMAs.
        
        Estrategia:
        - Compra (1): Cuando SMA corta cruza por encima de SMA larga
        - Venta (-1): Cuando SMA corta cruza por debajo de SMA larga
        - Mantener (0): En cualquier otro caso
        
        Args:
            short_period: Período de la SMA corta
            long_period: Período de la SMA larga
            
        Returns:
            DataFrame con señales añadidas
        """
        df_signals = self.df.copy()
        
        # Asegurar que las columnas existen
        short_sma = f'SMA_{short_period}'
        long_sma = f'SMA_{long_period}'
        
        if short_sma not in df_signals.columns or long_sma not in df_signals.columns:
            raise ValueError(f"Columnas {short_sma} o {long_sma} no encontradas en el DataFrame")
        
        # Crear columna para señales
        df_signals['signal'] = 0
        
        # Generar señales de compra (1) cuando SMA corta cruza por encima de SMA larga
        df_signals.loc[(df_signals[short_sma] > df_signals[long_sma]) & 
                      (df_signals[short_sma].shift(1) <= df_signals[long_sma].shift(1)), 'signal'] = 1
        
        # Generar señales de venta (-1) cuando SMA corta cruza por debajo de SMA larga
        df_signals.loc[(df_signals[short_sma] < df_signals[long_sma]) & 
                      (df_signals[short_sma].shift(1) >= df_signals[long_sma].shift(1)), 'signal'] = -1
        
        # Añadir columna de posición acumulada (para backtesting)
        df_signals['position'] = df_signals['signal'].cumsum()
        
        return df_signals
    
    def double_ema_crossover(self, short_period: int, long_period: int) -> pd.DataFrame:
        """
        Genera señales basadas en el cruce de dos EMAs.
        
        Estrategia:
        - Compra (1): Cuando EMA corta cruza por encima de EMA larga
        - Venta (-1): Cuando EMA corta cruza por debajo de EMA larga
        - Mantener (0): En cualquier otro caso
        
        Args:
            short_period: Período de la EMA corta
            long_period: Período de la EMA larga
            
        Returns:
            DataFrame con señales añadidas
        """
        df_signals = self.df.copy()
        
        # Asegurar que las columnas existen
        short_ema = f'EMA_{short_period}'
        long_ema = f'EMA_{long_period}'
        
        if short_ema not in df_signals.columns or long_ema not in df_signals.columns:
            raise ValueError(f"Columnas {short_ema} o {long_ema} no encontradas en el DataFrame")
        
        # Crear columna para señales
        df_signals['signal'] = 0
        
        # Generar señales de compra (1) cuando EMA corta cruza por encima de EMA larga
        df_signals.loc[(df_signals[short_ema] > df_signals[long_ema]) & 
                      (df_signals[short_ema].shift(1) <= df_signals[long_ema].shift(1)), 'signal'] = 1
        
        # Generar señales de venta (-1) cuando EMA corta cruza por debajo de EMA larga
        df_signals.loc[(df_signals[short_ema] < df_signals[long_ema]) & 
                      (df_signals[short_ema].shift(1) >= df_signals[long_ema].shift(1)), 'signal'] = -1
        
        # Añadir columna de posición acumulada (para backtesting)
        df_signals['position'] = df_signals['signal'].cumsum()
        
        return df_signals