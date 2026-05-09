import rasterio
from pathlib import Path
from pyproj import Transformer
import requests
import csv
import time
from rasterio.crs import CRS


def get_center_lon_lat(image_path):
    # Calcule le centre de l'image en longitude/latitude (WGS84)
    with rasterio.open(image_path) as src:
        left, bottom, right, top = src.bounds
        cx = (left + right) / 2
        cy = (bottom + top) / 2
        transformer = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(cx, cy)
        return lon, lat


def reverse_geocode_fr(lat, lon):
    # Interroge Nominatim en français et retourne (ville, pays).
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        'lat': lat,
        'lon': lon,
        'format': 'json',
        'accept-language': 'fr'  # Force la réponse en français
    }
    headers = {'User-Agent': 'SN7_geocoder/1.0'}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        address = data.get('address', {})
        # Recherche de la ville dans différents champs possibles
        ville = (address.get('city') or address.get('town') or
                 address.get('village') or address.get('hamlet') or
                 address.get('municipality') or '')
        pays = address.get('country', '')
        return ville, pays
    except Exception as e:
        print(f"Erreur géocodage ({lat}, {lon}) : {e}")
        return '', ''

def main():
    data_root = Path('data/train')
    output_dir = Path('outputs')
    output_file = output_dir / 'sites_ville_pays.csv'

    sites = [p for p in data_root.iterdir() if p.is_dir()]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['site', 'lon', 'lat', 'ville', 'pays'])

        for site in sites:
            img_dir = site / 'images'
            images = list(img_dir.glob('*.tif'))
            if not images:
                print(f"Aucune image trouvée pour {site.name}")
                writer.writerow([site.name, '', '', '', ''])
                continue

            try:
                lon, lat = get_center_lon_lat(images[0])
                print(f"{site.name} : {lon:.4f}, {lat:.4f} → ", end='')
                ville, pays = reverse_geocode_fr(lat, lon)
                print(f"{ville or '?'}, {pays or '?'}")
                writer.writerow([site.name, f"{lon:.6f}", f"{lat:.6f}", ville, pays])
                time.sleep(1)  # Respect des limites de Nominatim
            except Exception as e:
                print(f"Erreur pour {site.name} : {e}")
                writer.writerow([site.name, '', '', '', ''])
                

if __name__ == '__main__':
    main()