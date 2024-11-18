from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageTk
import tkinter as tk

# Carregar o modelo e os rótulos de classe
model = load_model('pneumonia_detection_model.h5')
class_labels = np.load('class_labels.npy', allow_pickle=True).item()

# Definir o limiar de confiança
CONFIDENCE_THRESHOLD = 0.5  # Ajuste conforme necessário

# Configurar a janela Tkinter
root = tk.Tk()
root.title("Pneumonia Detection")
root.geometry("600x400")

# Criar um label para exibir a imagem
label = tk.Label(root)
label.pack()

def select_image():
    # Abrir uma caixa de diálogo para selecionar o arquivo de imagem
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # Carregar e pré-processar a imagem
    img = Image.open(file_path).convert('RGB')  # Converte para RGB
    img = img.resize((128, 128))
    #img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalização

    # Fazer a predição da classe
    predictions = model.predict(img)
    confidence = predictions[0][0]  # Para classificação binária, temos um único valor de predição
    predicted_label = 'Pneumonia' if confidence >= CONFIDENCE_THRESHOLD else 'Normal'

    # Exibir o resultado na imagem
    result_text = f"Recognized: {predicted_label} (Confidence: {confidence:.2f})"
    img_display = Image.open(file_path).convert('RGB')
    img_display = img_display.resize((400, 350))
    frame = np.array(img_display)
    frame = cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Converter frame para ImageTk
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=image)
    
    # Atualizar o label com a nova imagem
    label.config(image=photo)
    label.image = photo

# Criar um botão para selecionar a imagem
button = tk.Button(root, text="Select Image", command=select_image)
button.pack()

# Iniciar o loop de eventos do Tkinter
root.mainloop()
