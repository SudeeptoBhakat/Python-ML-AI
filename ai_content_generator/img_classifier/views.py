from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
import os
from .forms import ImageForm
from .models import Image
from .utils import load_model_and_dict
import cv2
import numpy as np
from .utils import w2d

model, class_dict = load_model_and_dict()


@csrf_exempt
def index(request):
    return render(request, 'index.html')

# @csrf_exempt
def imageupload(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            saved_image = form.save()
            file_path = saved_image.photo.path

            img = cv2.imread(file_path)
            img = cv2.resize(img, (32, 32)) 
            features = extract_features(img)
            
            prediction = model.predict(features.reshape(1, -1))
            predicted_class = [key for key, value in class_dict.items() if value == prediction[0]][0]
            words = predicted_class.split('_')
            capitalized_name = [word.capitalize() for word in words]
            print(saved_image.photo.url)
            return JsonResponse({
                'success': True,
                'uploaded_file_url': saved_image.photo.url,
                'prediction': ' '.join(capitalized_name)
            })
        else:
            return JsonResponse({'success': False, 'errors': form.errors}, status=400)

    return render(request, 'index.html')

def extract_features(img):
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
    return combined_img