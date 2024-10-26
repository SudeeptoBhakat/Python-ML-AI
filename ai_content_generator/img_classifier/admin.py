from django.contrib import admin
from .models import Image, Query
# Register your models here.

@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'photo', 'date']

@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
     list_display = ['id', 'query_text', 'response_text', 'date', 'related_image']