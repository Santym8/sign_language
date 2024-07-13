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
from .forms import ImageForm
import io


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
        
        image = Image.open(image_data).convert('L')  # Convierte la imagen a escala de grises
        image = image.resize((28, 28))  # Ajusta el tamaño
        image = np.array(image) / 255.0  # Normaliza la imagen
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)

        # Realiza la predicción
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        predicted_sign = classes[predicted_class]

        return JsonResponse({'status': 'success', 'prediction': predicted_sign})
    except Exception as e:
        return JsonResponse({'status': 'error', 'error': str(e)})




def one_image(request):
    image_data_url = None
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            image_data = image.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            image_data_url = f"data:image/{image.content_type.split('/')[-1]};base64,{encoded_image}"

            image = Image.open(io.BytesIO(image_data))

            image = image.convert('L')  # Convierte la imagen a escala de grises
            image = image.resize((28, 28))  # Ajusta el tamaño
            image = np.array(image) / 255.0  # Normaliza la imagen
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_sign = classes[predicted_class]

            return render(request, 'one_image.html', {'form': form, 'image_data_url': image_data_url, 'predicted_sign': predicted_sign})
    else:
        form = ImageForm()
    return render(request, 'one_image.html', {'form': form, 'image_data_url': image_data_url})
