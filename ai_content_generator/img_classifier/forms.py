from django import forms
from .models import Image

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['photo']
        labels = {'photo': 'Upload Image'}
        widgets = {
            'photo': forms.ClearableFileInput(attrs={'accept': 'image/*'})
        }

    def clean_photo(self):
        photo = self.cleaned_data.get('photo')
        if not photo:
            raise forms.ValidationError("Please upload an image.")
        return photo
