from django import forms
from .models import Image, Query

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


class QueryForm(forms.ModelForm):
    class Meta:
        model = Query
        fields = ['query_text', 'related_image']
        labels = {
            'query_text': 'Enter your query',
            'related_image': 'Upload related image (optional)'
        }
        widgets = {
            'query_text': forms.Textarea(attrs={'placeholder': 'Type your query here...'}),
        }

    def clean_query_text(self):
        query_text = self.cleaned_data.get('query_text')
        if not query_text:
            raise forms.ValidationError("Query text cannot be empty.")
        return query_text