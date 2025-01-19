from django.contrib import admin
from .models import Party, Elections, ElectionData

admin.site.register(Party)
admin.site.register(Elections)
admin.site.register(ElectionData)