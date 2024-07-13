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


model_xception = load_model(os.path.join(settings.BASE_DIR, 'app/model/model_xception.h5'))
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

        # ----------------- Modelo Previo -----------------
        image = Image.open(image_data).convert('L')  # Convierte la imagen a escala de grises
        image = image.resize((28, 28))  # Ajusta el tamaño
        image = np.array(image) / 255.0  # Normaliza la imagen
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_sign = classes[predicted_class]

        # ----------------- Modelo Xception -----------------
        image_xception = Image.open(image_data).convert('RGB')
        image_xception = image_xception.resize((71, 71))
        image_xception = np.array(image_xception) / 255.0  # Normaliza la imagen
        image_xception = np.expand_dims(image_xception, axis=0)
        prediction_xception = model_xception.predict(image_xception)
        predicted_class_xception = np.argmax(prediction_xception, axis=1)[0]
        predicted_sign_xception = classes[predicted_class_xception]

        return JsonResponse({'status': 'success', 'prediction': predicted_sign, 'prediction_xception': predicted_sign_xception})
    except Exception as e:
        return JsonResponse({'status': 'error', 'error': str(e)})




def one_image(request):
    image_data_url = None
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_xception = request.FILES['image']
            image_data = image_xception.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            image_data_url = f"data:image/{image_xception.content_type.split('/')[-1]};base64,{encoded_image}"

            original_image = Image.open(io.BytesIO(image_data))

            # ----------------- Modelo Previo -----------------
            image = original_image.convert('L')  # Convierte la imagen a escala de grises
            image = image.resize((28, 28))  # Ajusta el tamaño
            image = np.array(image) / 255.0  # Normaliza la imagen
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_sign = classes[predicted_class]


            
            # ----------------- Modelo Xception -----------------
            image_xception = original_image.convert('RGB')
            image_xception = image_xception.resize((71, 71))
            image_xception = np.array(image_xception) / 255.0  # Normaliza la imagen
            image_xception = np.expand_dims(image_xception, axis=0)
            prediction_xception = model_xception.predict(image_xception)
            predicted_class_xception = np.argmax(prediction_xception, axis=1)[0]
            predicted_sign_xception = classes[predicted_class_xception]

            return render(request, 'one_image.html', {'form': form, 'image_data_url': image_data_url, 'predicted_sign_xception': predicted_sign_xception, 'predicted_sign': predicted_sign})
    else:
        form = ImageForm()
    return render(request, 'one_image.html', {'form': form, 'image_data_url': image_data_url})
