from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('<int:election_id>/<int:favoured_party_id>/', views.index, name='index'),
    #for '' redirect to '/1/0/'
    path('', views.redirect_to_index, name='redirect_to_index'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)