from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from datetime import datetime
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import SuspiciousOperation

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def save_photo(request):
    if request.method == 'POST':
        try:
            data = request.POST.get('image')
            if not data:
                raise SuspiciousOperation("No image data provided")
            
            format, imgstr = data.split(';base64,') 
            ext = format.split('/')[-1] 
            if ext not in ['jpg', 'jpeg', 'png']:
                raise SuspiciousOperation("Unsupported file format")
            
            img_data = base64.b64decode(imgstr)
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"
            filepath = os.path.join(settings.MEDIA_ROOT, 'img', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            return JsonResponse({'status': 'success', 'filename': filename})
        
        except (ValueError, SuspiciousOperation) as e:
            return JsonResponse({'status': 'failed', 'error': str(e)})
        except Exception as e:
            return JsonResponse({'status': 'failed', 'error': 'An error occurred while saving the image'})
    
    return JsonResponse({'status': 'failed', 'error': 'Invalid request method'})
