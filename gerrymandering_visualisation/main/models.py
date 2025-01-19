from django.db import models
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator
from django.utils.timezone import now
import re

def validate_colour(value):
    if not re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', value):
        raise ValidationError('%(value)s is not a valid HEX color.', params={'value': value})

def validate_geojson(value):
    if not value.name.endswith('.geojson'):
        raise ValidationError('Only .geojson files are allowed.')

class Party(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    colour = models.CharField(max_length=7, validators=[validate_colour])

    class Meta:
        verbose_name = 'Party'
        verbose_name_plural = 'Parties'

    def __str__(self):
        return self.name + ' (id: ' + str(self.id) + ')'

class Elections(models.Model):
    id = models.AutoField(primary_key=True)
    year = models.IntegerField(validators=[MaxValueValidator(now().year)])
    parties = models.ManyToManyField('Party', related_name='elections')
    real_results = models.JSONField() #real_results = {party_id: votes}
    real_seats = models.JSONField() #real_seats = {party_id: seats}
    # Example JSON for real_results
    # real_results = {"1": 5000, "2": 3000, "3": 2000}
    # This means Party with id 1 got 5000 votes, Party with id 2 got 3000 votes, and Party with id 3 got 2000 votes.

    # Example JSON for real_seats
    # real_seats = {"1": 5, "2": 3, "3": 2}
    # This means Party with id 1 won 5 seats, Party with id 2 won 3 seats, and Party with id 3 won 2 seats.

    class Meta:
        ordering = ['-year']
        verbose_name = 'Elections'
        verbose_name_plural = 'Elections'

    def __str__(self):
        return str(self.year) + ' (id: ' + str(self.id) + ')'

class ElectionData(models.Model):
    elections = models.ForeignKey('Elections', on_delete=models.CASCADE)
    data_file = models.FileField(upload_to='election_data/', validators=[validate_geojson])
    favoured_party = models.ForeignKey('Party', on_delete=models.SET_NULL, null=True, related_name='favoured_data', blank=True)
    seats = models.JSONField(null=True) #seats = {party_id: seats}

    class Meta:
        unique_together = ('elections', 'favoured_party')
        verbose_name = 'Election Data'
        verbose_name_plural = 'Election Data'

    def __str__(self):
        return f'{self.elections.year} - {self.favoured_party}' if self.favoured_party else f'{self.elections.year} - No favoured party'
