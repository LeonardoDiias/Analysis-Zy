import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model # type: ignore

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Carregamento do Model / Class #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

model = load_model('pneumonia_detection_model.h5') #('pneumonia_detection_model.keras') o novo caminho apos um novo treinamento...
class_label = np.load('class_labels.npy', allow_pickle=True).item()

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Decisão dos Dados #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

COFIDENCE_THRESHOLD = 0.5

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=# Configuração do Tkinter #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

root = tk.Tk()
root.title("Zy Analyzer")
root.geometry("600x420")
label = tk.Label(root)
label.pack()
label_txt = tk.Label(root, text="")
label_txt.pack()

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
    
    #=#=#=#=#=#=#=#=#  Predição  da  Classe  #=#=#=#=#=#=#=#=#

    previsao = model.predict(image)
    confianca = previsao[0][0]
    previsao_label = 'Pneumonia' if confianca >= COFIDENCE_THRESHOLD else 'Normal'
    
    #=#=#=#=#=#=#=#=# Exibição de Imagem #=#=#=#=#=#=#=#=#
    image_display = Image.open(file_path).convert('RGB').resize((400,350))
    frame = np.array(image_display)
    #frama_RGB = cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    img = Image.fromarray(frame)
    imagem = ImageTk.PhotoImage(image=img)

    #=#=#=#=#=#=#=#=# Atualização da Imagem #=#=#=#=#=#=#=#=#
    label.config(image=imagem)
    label.image = imagem

    #=#=#=#=#=#=#=#=# Exibição de Resultado #=#=#=#=#=#=#=#=#
    resultado_txt = f"Result: {previsao_label}"
    label_txt.config(text=resultado_txt)

#=#=#=#=#=#=#=#=# Criação do Botão #=#=#=#=#=#=#=#=#
# Criar um botão estilizado para selecionar a imagem
button = tk.Button(root, text="Selecione a Imagem", command=selection_image, 
                   bg='lightblue', fg='black', font=('Sans Serif', 12, 'bold'), 
                   relief=tk.RAISED, bd=2, padx=10, pady=5)
button.pack()


root.mainloop()