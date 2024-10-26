from django.db import models

# Create your models here.
class Image(models.Model):
    photo = models.ImageField(upload_to="uploads")
    date = models.DateTimeField(auto_now_add=True)

class Query(models.Model):
    query_text = models.TextField()
    response_text = models.TextField()
    date = models.DateTimeField(auto_now_add=True)
    related_image = models.ForeignKey(Image, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"Query on {self.date}: {self.query_text[:50]}..."