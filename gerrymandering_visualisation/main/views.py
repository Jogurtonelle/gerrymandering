import json
from django.shortcuts import render, get_object_or_404
from django.http import Http404
import os

from .models import Party, Elections, ElectionData

def index(request, election_id, favoured_party_id):
    try:
        # Pobranie obiektu wyborów na podstawie roku
        election = get_object_or_404(Elections, id=election_id)

        # Pobranie obiektu partii na podstawie ID
        if favoured_party_id == 0:
            favoured_party = None
        else:
            favoured_party = get_object_or_404(Party, id=favoured_party_id)

        # Pobranie danych wyborczych związanych z tymi wyborami i faworyzowaną partią
        election_data = ElectionData.objects.get(elections=election, favoured_party=favoured_party)

        # Ścieżka do pliku geojson na podstawie ścieżki zapisanej w `election_data`
        geojson_path = election_data.data_file.path

        # Sprawdzenie, czy plik geojson istnieje
        if not os.path.exists(geojson_path):
            raise Http404("Brak pliku geojson")
        
        geojson_content = open(geojson_path).read()

        elections = Elections.objects.all()

        elections_to_parties = {}
        for election in Elections.objects.all():
            parties = election.parties.all()
            elections_to_parties[election.id] = [
                {'id': party.id, 'name': party.name} for party in parties
            ]

        party_colours = {party.id: party.colour for party in Party.objects.all() }

        real_results = election.real_results
        real_seats = election.real_seats
        seats = election_data.seats
        party_id_to_name = {party.id: party.name for party in get_object_or_404(Elections, id=election_id).parties.all()}

        context = {
            'election': election,
            'favoured_party': favoured_party,
            'geojson_content': geojson_content,
            'elections': elections,
            'elections_to_parties': elections_to_parties,
            'party_colours': party_colours,
            'real_results': real_results,
            'real_seats': real_seats,
            'seats': seats,
            'party_id_to_name': party_id_to_name,
        }


        return render(request, 'index.html', context)

    except ElectionData.DoesNotExist:
        raise Http404("Brak danych wyborczych dla podanych wyborów i partii")