# Torres Condor Gerald Luis
# 2024200534A
# Tema 44: Determinantes del indice S&P/BVL Peru General, 2015-2025
# Fecha de limpieza: 2026-09-25

import os
import hashlib
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    filename='../log_ejecucion.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CODIGO_MATRICULA = "2024200534A"
CARPETA_CRUDOS = "../datos_crudos"
CARPETA_PROCESADOS = "../datos_procesados"
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

# Mapeo de meses en espanol
MESES = {
    'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
}


def parsear_fecha_bcrp(texto):
    """Convierte 'Ene.2015' a '2015-01-01'."""
    try:
        mes, anio = texto.replace('.', ' ').split()
        return pd.Timestamp(f"{anio}-{MESES[mes]}-01")
    except Exception:
        return pd.NaT


def cargar_indice():
    ruta = f"{CARPETA_CRUDOS}/bcrp_indice_general_bvl_{CODIGO_MATRICULA}.csv"
    df = pd.read_csv(ruta)
    df['fecha'] = df['fecha'].apply(parsear_fecha_bcrp)
    df = df.dropna(subset=['fecha'])
    df = df.sort_values('fecha')
    logging.info(f"Indice cargado: {len(df)} filas mensuales")
    return df


def cargar_yahoo_mensual(nombre):
    """Carga serie diaria de Yahoo y la agrega a mensual (promedio)."""
    ruta = f"{CARPETA_CRUDOS}/yahoo_{nombre}_{CODIGO_MATRICULA}.csv"
    df = pd.read_csv(ruta, parse_dates=['fecha'])
    df = df.set_index('fecha')
    df_mensual = df.resample('MS').mean().reset_index()
    df_mensual = df_mensual.rename(columns={'fecha': 'fecha_mes'})
    logging.info(f"{nombre} cargado y agregado a mensual: {len(df_mensual)} filas")
    return df_mensual


def construir_procesado():
    logging.info("=== INICIO LIMPIEZA ===")

    df_indice = cargar_indice()
    df_tc = cargar_yahoo_mensual('tipo_cambio')
    df_cobre = cargar_yahoo_mensual('precio_cobre')

    # Unir por fecha mensual
    df = df_indice.merge(df_tc, left_on='fecha', right_on='fecha_mes', how='inner')
    df = df.merge(df_cobre, left_on='fecha', right_on='fecha_mes', how='inner')
    df = df.drop(columns=['fecha_mes_x', 'fecha_mes_y'], errors='ignore')

    # Renombrar columnas
    df = df.rename(columns={'indice_general_bvl': 'indice_bvl'})

    # Ordenar y resetear indice
    df = df.sort_values('fecha').reset_index(drop=True)

    # Eliminar filas con NaN
    df = df.dropna()

    logging.info(f"Dataset unificado: {len(df)} filas, {len(df.columns)} columnas")
    logging.info(f"Columnas: {list(df.columns)}")
    logging.info(f"Rango: {df['fecha'].min()} a {df['fecha'].max()}")

    ruta = f"{CARPETA_PROCESADOS}/datos_procesados_{CODIGO_MATRICULA}.csv"
    df.to_csv(ruta, index=False)
    logging.info(f"Guardado: {ruta}")

    # Hash SHA-256
    with open(ruta, 'rb') as f:
        hash_sha256 = hashlib.sha256(f.read()).hexdigest()
    logging.info(f"Hash SHA-256: {hash_sha256}")

    # Escribir hash en un archivo aparte para el README
    with open(f"{CARPETA_PROCESADOS}/hash_sha256.txt", 'w') as f:
        f.write(hash_sha256)

    return df, hash_sha256


if __name__ == "__main__":
    df, hash_val = construir_procesado()
    print(f"Dataset final: {len(df)} filas, {len(df.columns)} columnas")
    print(f"Hash SHA-256: {hash_val}")
    print(df.head())
