import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Carregamento do Model / Class #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

model = load_model('pneumonia_detection_model.keras')
class_label = np.load('class_labels.npy', allow_pickle=True).item()

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Decisão dos Dados #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

COFIDENCE_THRESHOLD = 0.5

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Configuração do Tkinter #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

root = tk.Tk()
root.title("Pneumonia Detection")
root.geometry("600x400")
label = tk.Label(root)
label.pack()

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Painel de Pre-processamento #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

def selection_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    #=#=#=#=#=#=#=#=# Processamento de Imagem #=#=#=#=#=#=#=#=#
    
    image = Image.open(file_path).convert('RGB')
    image = image.resize((128,128))
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    #=#=#=#=#=#=#=#=#  Previsão  da  Classe  #=#=#=#=#=#=#=#=#

    previsao = model.predict(image)
    confianca = previsao[0][0]
    previsao_label = 'Pneumonia' if confianca >= COFIDENCE_THRESHOLD else 'Normal'
    
    #=#=#=#=#=#=#=#=# Exibição da Imagem*Resultado #=#=#=#=#=#
