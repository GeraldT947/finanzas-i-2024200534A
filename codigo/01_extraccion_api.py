# Torres Condor Gerald Luis
# 2024200534A
# Tema 44: Determinantes del indice S&P/BVL Peru General, 2015-2025
# Fecha de extraccion: 2026-09-28

import os
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

logging.basicConfig(
    filename='../log_ejecucion.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

FECHA_INICIO = "2015-1"
FECHA_CORTE  = "2024-12"
CODIGO_MATRICULA = "2024200534A"
CARPETA_CRUDOS = "../datos_crudos"
os.makedirs(CARPETA_CRUDOS, exist_ok=True)


def extraer_serie_bcrp(nombre, codigo):
    url = (f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
           f"{codigo}/json/{FECHA_INICIO}/{FECHA_CORTE}")
    logging.info(f"BCRP: {nombre} ({codigo})")
    try:
        resp = requests.get(url, timeout=60)
        logging.info(f"HTTP {resp.status_code} para {nombre}")
        resp.raise_for_status()
        data = resp.json()
        if "periods" not in data:
            logging.warning(f"Sin periods para {nombre}")
            return pd.DataFrame()
        registros = []
        for p in data["periods"]:
            valores = p.get("values", [])
            valor = None
            if valores and valores[0] not in ["n.d.", None, ""]:
                try:
                    valor = float(str(valores[0]).replace(",", ""))
                except (ValueError, TypeError):
                    valor = None
            registros.append({"fecha": p.get("name"), "valor": valor})
        df = pd.DataFrame(registros)
        df = df[df["valor"].notna()]
        df = df.rename(columns={"valor": nombre})
        ruta = f"{CARPETA_CRUDOS}/bcrp_{nombre}_{CODIGO_MATRICULA}.csv"
        df.to_csv(ruta, index=False)
        logging.info(f"Guardado: {ruta} ({len(df)} filas)")
        return df
    except Exception as e:
        logging.error(f"Error en {nombre}: {e}")
        return pd.DataFrame()


def extraer_yahoo(ticker, nombre):
    logging.info(f"Yahoo: {nombre} ({ticker})")
    try:
        df = yf.download(ticker, start="2015-01-01", end="2024-12-31", progress=False)
        if df.empty:
            logging.warning(f"Sin datos para {nombre}")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if 'date' not in df.columns or 'close' not in df.columns:
            logging.error(f"Columnas inesperadas en {nombre}")
            return pd.DataFrame()
        df = df[['date', 'close']].rename(columns={'date': 'fecha', 'close': nombre})
        df['fecha'] = pd.to_datetime(df['fecha']).dt.date
        df[nombre] = pd.to_numeric(df[nombre], errors='coerce')
        ruta = f"{CARPETA_CRUDOS}/yahoo_{nombre}_{CODIGO_MATRICULA}.csv"
        df.to_csv(ruta, index=False)
        logging.info(f"Guardado: {ruta} ({len(df)} filas)")
        return df
    except Exception as e:
        logging.error(f"Error Yahoo {nombre}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    logging.info("========== INICIO ==========")
    inicio = datetime.now()

    # BCRP: 4 series
    extraer_serie_bcrp("indice_general_bvl", "PN01142MM")
    extraer_serie_bcrp("tasa_referencia", "PD04722MM")
    extraer_serie_bcrp("ipc", "PN01273PM")

    # Yahoo Finance: 2 series
    extraer_yahoo("PEN=X", "tipo_cambio")
    extraer_yahoo("HG=F", "precio_cobre")

    fin = datetime.now()
    logging.info(f"Duracion: {fin - inicio}")
    logging.info("========== FIN ==========")
    print("Extraccion completada.")
