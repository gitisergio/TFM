# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:10:40 2022

@author: Sergio
"""
import pandas as pd
import joblib
import sys

def predecir(df,loaded_model):
    pred_prob = loaded_model.predict_proba(df)
    return pred_prob

def resultados_dicc(pred_prob,asignaturas):    
    resultados = []
    for i in pred_prob:
        dicc_resultados = {}
        for j in range(len(i)):
           dicc_resultados[asignaturas[j]] = i[j]
        resultados.append(dicc_resultados)
    return resultados
        
def resultados_ord(resultados):
    resultados_ord = []
    for i in resultados:
        dicc_ordenado = dict(sorted(i.items(), key=lambda item: item[1],reverse=True))
        dicc_ordenado = {k: dicc_ordenado[k] for k in list(dicc_ordenado)[:3]}
        resultados_ord.append(dicc_ordenado)
    return resultados_ord



def main(dicc):
    asignaturas = {0: 'Administración de Negocios Internacionales',
                   1: 'Comunicación Social',
                   2: 'Contaduría Pública',
                   3: 'Derecho',
                   4: 'Enfermería',
                   5: 'Fisioterapia',
                   6: 'Ingeniería Ambiental',
                   7: 'Ingeniería Civil',
                   8: 'Ingeniería Mecatrónica',
                   9: 'Ingeniería de Procesos',
                   10: 'Ingeniería de Sistemas',
                   11: 'Licenciatura en Educación Básica Primaria',
                   12: 'Licenciatura en Educación Infantil',
                   13: 'Mercadeo',
                   14: 'Nutrición y Dietética',
                   15: 'Psicología',
                   16: 'Tecnología en Radiodiagnóstico y Radioterapia',
                   17: 'Tecnología en Regencia de Farmacia',
                   18: 'Terapia Ocupacional',
                   19: 'Trabajo Social',
                   20: 'Técnico Laboral en Auxiliar en Enfermería'}
    
    filename = 'Esparta.sav'
    loaded_model = joblib.load(filename)
    
    filename = 'Code.sav'
    encoder = joblib.load(filename)
    
    p_df = pd.DataFrame(dicc)
    columns = p_df.columns
    encoded_cols_test = encoder.transform(p_df[columns]).toarray()
    p_df_transf = pd.DataFrame(encoded_cols_test)

    pred_prob = predecir(p_df_transf,loaded_model)
    resultados = resultados_dicc(pred_prob,asignaturas)
    k = resultados_ord(resultados)
    print(k)
    
    return k

if __name__ == "__main__":
    ingles = sys.argv[5]
    sociales = sys.argv[3]
    mates = sys.argv[1]
    naturales = sys.argv[4]
    lectura = sys.argv[2]
    diccionario = {'INGLÉS':[ingles],'CIENCIAS SOCIALES':[sociales],'MATEMÁTICAS':[mates],'CIENCIAS NATURALES':[naturales],'LECTURA CRÍTICA':[lectura]}
    main(diccionario)
















