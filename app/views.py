from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from datetime import datetime
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import SuspiciousOperation
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo preentrenado (asegúrate de que el archivo .h5 está en la ruta correcta)
model = load_model(os.path.join(settings.BASE_DIR, 'app/model/model.h5'))

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def predict_hand_sign(request):
    if request.method != 'POST':
        raise SuspiciousOperation("Invalid request method")

    image_data = request.POST.get('image')
    if not image_data:
        return JsonResponse({'status': 'error', 'error': 'No image data provided'})

    try:
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        image_data = ContentFile(base64.b64decode(imgstr), name=f'hand_image_{datetime.now().strftime("%Y%m%d%H%M%S")}.{ext}')
        
         # Abre la imagen usando PIL
        image = Image.open(image_data).convert('L')  # Convierte la imagen a escala de grises
        image = image.resize((28, 28))  # Ajusta el tamaño según lo requiera tu modelo
        image = np.array(image) / 255.0  # Normaliza la imagen
        image = np.expand_dims(image, axis=-1)  # Añade una dimensión para el canal
        image = np.expand_dims(image, axis=0)  # Añade una dimensión para el batch

        # Realiza la predicción
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Mapea la clase predicha a un signo de mano (ajusta esto según tu modelo)
        predicted_sign = classes[predicted_class]

        return JsonResponse({'status': 'success', 'prediction': predicted_sign})
    except Exception as e:
        return JsonResponse({'status': 'error', 'error': str(e)})
