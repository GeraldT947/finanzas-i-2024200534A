# Torres Condor Gerald Luis
# 2024200534A
# Tema 44: Determinantes del indice S&P/BVL Peru General, 2015-2025
# Fecha de limpieza: 2026-09-28

import os
import hashlib
import logging
import pandas as pd

BASE = "/content/drive/MyDrive/finanzas-i-2024200534A"

logging.basicConfig(
    filename=f'{BASE}/log_ejecucion.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CODIGO_MATRICULA = "2024200534A"
CARPETA_CRUDOS = f"{BASE}/datos_crudos"
CARPETA_PROCESADOS = f"{BASE}/datos_procesados"
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

MESES = {
    'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
}


def parsear_fecha_bcrp(texto):
    try:
        partes = texto.replace('.', ' ').split()
        if len(partes) == 2:
            mes, anio = partes
            return pd.Timestamp(f"{anio}-{MESES[mes]}-01")
        return pd.NaT
    except Exception:
        return pd.NaT


def cargar_serie_bcrp(nombre):
    ruta = f"{CARPETA_CRUDOS}/bcrp_{nombre}_{CODIGO_MATRICULA}.csv"
    df = pd.read_csv(ruta)
    df['fecha'] = df['fecha'].apply(parsear_fecha_bcrp)
    df = df.dropna(subset=['fecha']).sort_values('fecha')
    logging.info(f"{nombre} cargado: {len(df)} filas")
    return df


def cargar_yahoo_mensual(nombre):
    ruta = f"{CARPETA_CRUDOS}/yahoo_{nombre}_{CODIGO_MATRICULA}.csv"
    df = pd.read_csv(ruta, parse_dates=['fecha'])
    df = df.set_index('fecha')
    df_mensual = df.resample('MS').mean().reset_index()
    df_mensual = df_mensual.rename(columns={'fecha': 'fecha'})
    logging.info(f"{nombre} agregado a mensual: {len(df_mensual)} filas")
    return df_mensual


def construir_procesado():
    logging.info("=== INICIO LIMPIEZA ===")

    df_indice = cargar_serie_bcrp("indice_general_bvl")
    df_tasa = cargar_serie_bcrp("tasa_referencia")
    df_ipc = cargar_serie_bcrp("ipc")
    df_tc = cargar_yahoo_mensual("tipo_cambio")
    df_cobre = cargar_yahoo_mensual("precio_cobre")

    # Unir todo por fecha
    df = df_indice.merge(df_tc, on='fecha', how='inner')
    df = df.merge(df_cobre, on='fecha', how='inner')
    df = df.merge(df_tasa, on='fecha', how='inner')
    df = df.merge(df_ipc, on='fecha', how='inner')

    # Ordenar
    df = df.sort_values('fecha').reset_index(drop=True)
    df = df.dropna()

    # Renombrar columna del indice
    df = df.rename(columns={'indice_general_bvl': 'indice_bvl'})

    logging.info(f"Dataset unificado: {len(df)} filas, {len(df.columns)} columnas")
    logging.info(f"Columnas: {list(df.columns)}")

    ruta = f"{CARPETA_PROCESADOS}/datos_procesados_{CODIGO_MATRICULA}.csv"
    df.to_csv(ruta, index=False)

    with open(ruta, 'rb') as f:
        hash_sha256 = hashlib.sha256(f.read()).hexdigest()
    logging.info(f"Hash SHA-256: {hash_sha256}")

    with open(f"{CARPETA_PROCESADOS}/hash_sha256.txt", 'w') as f:
        f.write(hash_sha256)

    return df, hash_sha256


if __name__ == "__main__":
    df, hash_val = construir_procesado()
    print(f"Dataset final: {len(df)} filas, {len(df.columns)} columnas")
    print(f"Columnas: {list(df.columns)}")
    print(f"Hash SHA-256: {hash_val}")
    print(df.head())
