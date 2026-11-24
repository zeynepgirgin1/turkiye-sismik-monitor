import asyncio
import io
import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

# ==========================================
# 1. PARAMETRELER
# ==========================================
MIN_LAT, MAX_LAT = 35.5, 42.5
MIN_LON, MAX_LON = 25.5, 45.0
HEX_RADIUS_DEG = 0.28     # Altıgen yarıçapı (~30 km)
SEARCH_RADIUS_KM = 45.0
MC = 2.6
DELTA_M = 0.1
RECENT_DAYS = 30
MIN_EVENTS_FOR_B = 10
REFRESH_INTERVAL_SEC = 300  # 5 dakika

LIVE_STATE = {
    "updated_at": None,
    "next_update_sec": REFRESH_INTERVAL_SEC,
    "hexagons": [],
    "cities": [],
    "geothermal": [],
    "recent_quakes": [],
}

HISTORICAL_CACHE = {"df_50y": None, "df_5y_base": None}


# ==========================================
# 2. JEOTERMAL HAVZALAR & 81 İL + İLÇE ZEMİN (Vs30) VERİSİ
# ==========================================
GEOTHERMAL_ZONES = [
    {"name": "Denizli - Kızıldere / Büyük Menderes", "lat": 37.92, "lon": 28.82, "radius_km": 55, "temp_c": "200–242°C", "type": "Yüksek Entalpili Graben Sistemi"},
    {"name": "Aydın - Germencik / Salavatlı", "lat": 37.87, "lon": 27.60, "radius_km": 45, "temp_c": "180–232°C", "type": "Büyük Menderes Rift Zonu"},
    {"name": "Manisa - Salihli / Alaşehir (Gediz)", "lat": 38.48, "lon": 28.15, "radius_km": 50, "temp_c": "160–215°C", "type": "Gediz Grabeni Hidrotermal"},
    {"name": "Kütahya - Simav / Emet", "lat": 39.12, "lon": 28.95, "radius_km": 45, "temp_c": "140–165°C", "type": "Simav Fayı Akışkan Zonu"},
    {"name": "İzmir - Balçova / Seferihisar", "lat": 38.32, "lon": 26.95, "radius_km": 40, "temp_c": "135–155°C", "type": "Tuzla-Seferihisar Fay Zonu"},
    {"name": "Çanakkale - Tuzla / Kestanbol", "lat": 39.62, "lon": 26.35, "radius_km": 40, "temp_c": "145–174°C", "type": "Biga Yarımadası Jeotermal"},
    {"name": "Balıkesir - Gönen / Edremit", "lat": 39.85, "lon": 27.35, "radius_km": 45, "temp_c": "80–115°C", "type": "Kazdağı-Gönen Kırık Sistemi"},
    {"name": "Afyonkarahisar - Sandıklı / Ömer-Gecek", "lat": 38.78, "lon": 30.45, "radius_km": 45, "temp_c": "95–110°C", "type": "Akşehir-Simav Fay Sistemi"},
    {"name": "Ankara - Kızılcahamam / Beypazarı", "lat": 40.48, "lon": 32.65, "radius_km": 40, "temp_c": "75–88°C", "type": "Galatya Volkanik Masifi"},
    {"name": "Nevşehir - Kozaklı / Acıgöl", "lat": 38.85, "lon": 34.75, "radius_km": 45, "temp_c": "90–140°C", "type": "Kapadokya Volkanik Kompleksi"},
    {"name": "Bitlis - Nemrut / Süphan Zonu", "lat": 38.65, "lon": 42.22, "radius_km": 50, "temp_c": "120–180°C", "type": "Genç Volkanik / Hidrotermal"},
    {"name": "Ağrı - Diyadin / Tendürek", "lat": 39.52, "lon": 43.95, "radius_km": 45, "temp_c": "85–130°C", "type": "Tendürek Volkanik Sistemi"},
]

CITIES_DATA = [
    ("İstanbul", "Zeytinburnu", 40.994, 28.904, "Alüvyon / Killi Kireçtaşı (ZD-ZE)", 210, "KAF - Marmara Segmenti", 24.0),
    ("İstanbul", "Avcılar", 40.979, 28.721, "Gevşek Kil / Heyelanlı Zemin (ZE)", 195, "KAF - Marmara Segmenti", 24.0),
    ("İstanbul", "Bakırköy", 40.980, 28.872, "Güngören/Bakırköy Formasyonu (ZD)", 235, "KAF - Marmara Segmenti", 24.0),
    ("İstanbul", "Kadıköy", 40.981, 29.028, "Kireçtaşı / Kıyı Alüvyonu (ZC)", 410, "KAF - Adalar Segmenti", 24.0),
    ("İstanbul", "Şişli", 41.060, 28.987, "Trakya Formasyonu / Sağlam Kaya (ZB)", 680, "KAF - Marmara Segmenti", 24.0),
    ("İstanbul", "Sarıyer", 41.166, 29.050, "Paleozoyik Kayaç / Grovak (ZB)", 750, "KAF - Kuzey Kuşağı", 22.0),
    ("İstanbul", "Tuzla", 40.816, 29.300, "Kıyı Dolgusu / Alüvyon (ZD)", 245, "KAF - Çınarcık/İzmit", 24.0),
    ("İzmir", "Bayraklı", 38.462, 27.166, "Kalın Havza Alüvyonu (ZE)", 180, "İzmir Fayı / Gediz Grabeni", 18.0),
    ("İzmir", "Bornova (Kaya)", 38.471, 27.222, "Kireçtaşı / Andezit Kaya (ZB)", 620, "İzmir Fayı / Gediz Grabeni", 18.0),
    ("İzmir", "Seferihisar", 38.196, 26.838, "Alüvyon / Jeotermal Zon (ZD)", 260, "Tuzla-Seferihisar Fayı", 19.0),
    ("Adana", "Seyhan", 36.991, 35.330, "Çukurova Delta Alüvyonu (ZD-ZE)", 220, "Kıbrıs Yayı / Karataş-Osmaniye", 10.0),
    ("Adıyaman", "Merkez", 37.764, 38.278, "Havza Çökelleri / Kireçtaşı (ZC)", 340, "Doğu Anadolu Fay Zonu", 11.0),
    ("Afyonkarahisar", "Merkez", 38.756, 30.538, "Akarçay Havza Alüvyonu (ZD)", 240, "Akşehir-Simav Fay Sistemi", 14.0),
    ("Ağrı", "Merkez", 39.719, 43.050, "Alüvyon / Volkanik Tüf (ZD)", 255, "Tutak / Doğubayazıt Fayı", 10.0),
    ("Aksaray", "Merkez", 38.368, 34.036, "Tuz Gölü Havza Çökeli (ZC)", 330, "Tuz Gölü Fay Zonu", 5.0),
    ("Amasya", "Merkez", 40.653, 35.833, "Yeşilırmak Vadisi Alüvyonu (ZD)", 250, "Kuzey Anadolu Fay Zonu (KAF)", 24.0),
    ("Ankara", "Çankaya", 39.918, 32.854, "Andezit / Sağlam Kaya (ZB)", 690, "Orta Anadolu İkincil", 4.0),
    ("Antalya", "Muratpaşa", 36.884, 30.705, "Traverten Platosu (ZB-ZC)", 560, "Helen-Kıbrıs Dalma Zonu", 25.0),
    ("Ardahan", "Merkez", 41.110, 42.702, "Bazalt Platosu / Ayrışmış Kaya (ZC)", 440, "Kuzeydoğu Anadolu Fayı", 8.0),
    ("Artvin", "Merkez", 41.182, 41.818, "Volkanik / Yamaç Kayacı (ZB)", 640, "Karadeniz / Doğu Pontid", 6.0),
    ("Aydın", "Efeler", 37.844, 27.845, "Büyük Menderes Alüvyonu (ZE)", 205, "Büyük Menderes Grabeni", 18.0),
    ("Balıkesir", "Altıeylül", 39.648, 27.882, "Balıkesir Ovası Alüvyonu (ZD)", 240, "Yenice-Gönen / Havran-Balıkesir", 18.0),
    ("Bartın", "Merkez", 41.634, 32.337, "Kireçtaşı / Kıyı Alüvyonu (ZC)", 410, "Bartın Ters Fayı", 6.0),
    ("Batman", "Merkez", 37.881, 41.135, "Dicle Havza Çökelleri (ZC)", 350, "Bitlis-Zagros Kenet Kuşağı", 9.0),
    ("Bayburt", "Merkez", 40.255, 40.224, "Kireçtaşı / Volkanik Kaya (ZC)", 460, "Aşkale / KAF Güney Kolu", 12.0),
    ("Bilecik", "Merkez", 40.142, 29.979, "Metamorfik / Kireçtaşı Kaya (ZB)", 610, "KAF Güney Kolu / Eskişehir Fayı", 12.0),
    ("Bingöl", "Merkez", 38.885, 40.498, "Genç Havza Dolgusu (ZD)", 260, "KAF-DAF Üçlü Eklemi", 20.0),
    ("Bitlis", "Merkez", 38.400, 42.109, "Volkanik Tüf / Metamorfik (ZC)", 430, "Bitlis-Zagros / Nemrut Zonu", 10.0),
    ("Bolu", "Merkez", 40.735, 31.606, "Bolu Ovası Kalın Alüvyon (ZE)", 205, "Kuzey Anadolu Fay Zonu (KAF)", 24.0),
    ("Burdur", "Merkez", 37.720, 30.290, "Burdur Gölü Çökelleri (ZD)", 230, "Fethiye-Burdur Fay Zonu", 16.0),
    ("Bursa", "Nilüfer", 40.218, 28.985, "Bursa Ovası Alüvyonu (ZD)", 230, "Uluabat / Bursa Fayı", 16.0),
    ("Çanakkale", "Merkez", 40.155, 26.414, "Kıyı Alüvyonu / Kumtaşı (ZD)", 250, "KAF Kuzey/Güney Kolu", 22.0),
    ("Çankırı", "Merkez", 40.601, 33.613, "Jipsli Seri / Havza Çökeli (ZC)", 320, "KAF / Çankırı Fayı", 18.0),
    ("Çorum", "Merkez", 40.550, 34.955, "Havza Alüvyonu / Kiltaşı (ZC)", 340, "KAF / Sungurlu-Ezinepazarı", 18.0),
    ("Denizli", "Merkezefendi", 37.783, 29.085, "Büyük Menderes Alüvyonu (ZD)", 235, "Pamukkale / Babadağ Fayı", 18.0),
    ("Diyarbakır", "Kayapınar", 37.914, 40.230, "Bazalt Platosu / Kil (ZC)", 450, "Bitlis-Zagros Kenet Kuşağı", 9.0),
    ("Düzce", "Merkez", 40.843, 31.156, "Düzce Ovası Kalın Alüvyon (ZE)", 190, "KAF - Düzce Segmenti", 24.0),
    ("Edirne", "Merkez", 41.677, 26.555, "Ergene Havzası Çökelleri (ZD)", 265, "Trakya Havzası (Düşük Tektonik)", 4.0),
    ("Elazığ", "Merkez", 38.674, 39.222, "Ayrışmış Kaya / Alüvyon (ZC)", 390, "DAF - Palu/Hazar Segmenti", 11.0),
    ("Erzincan", "Merkez", 39.750, 39.491, "Erzincan Havzası Kalın Alüvyon (ZE)", 205, "KAF - Yedisu/Erzincan", 24.0),
    ("Erzurum", "Yakutiye", 39.904, 41.267, "Erzurum Ovası Alüvyonu (ZD)", 250, "Aşkale / Dumlu Fay Zonu", 12.0),
    ("Eskişehir", "Odunpazarı", 39.776, 30.520, "Porsuk Alüvyonu / Kaya Geçişi (ZC)", 310, "Eskişehir Fay Zonu", 8.0),
    ("Gaziantep", "Şahinbey", 37.066, 37.383, "Kireçtaşı / Marn (ZC)", 470, "Ölü Deniz / DAF Etkisi", 9.0),
    ("Giresun", "Merkez", 40.912, 38.389, "Volkanik Kaya / Kıyı Dolgusu (ZC)", 490, "Doğu Karadeniz Kıyı Kuşağı", 4.0),
    ("Gümüşhane", "Merkez", 40.460, 39.481, "Granit / Volkanik Kaya (ZB)", 650, "KAF Kuzey Etkisi", 8.0),
    ("Hakkari", "Merkez", 37.574, 43.740, "Kireçtaşı / Sıkışma Zonu Kaya (ZB-ZC)", 510, "Şemdinli-Yüksekova Fayı", 12.0),
    ("Hatay", "Antakya", 36.202, 36.160, "Asi Nehri Alüvyonu (ZE)", 195, "Ölü Deniz / DAF Güney Ucu", 11.0),
    ("Iğdır", "Merkez", 39.916, 44.045, "Iğdır Ovası Kalın Alüvyon (ZE)", 200, "Iğdır / Doğubayazıt Fayı", 10.0),
    ("Isparta", "Merkez", 37.764, 30.556, "Havza Çökelleri / Volkanik (ZC)", 320, "Isparta Açısı / Burdur Fayı", 14.0),
    ("Kahramanmaraş", "Onikişubat", 37.585, 36.922, "Havza Çökelleri (ZC-ZD)", 310, "Doğu Anadolu Fay Zonu", 11.0),
    ("Karabük", "Merkez", 41.206, 32.620, "Kireçtaşı / Vadide Alüvyon (ZC)", 420, "KAF Kuzey Etkisi", 12.0),
    ("Karaman", "Merkez", 37.175, 33.228, "Konya-Karaman Göl Çökeli (ZC)", 340, "Ecemiş / Orta Anadolu", 4.0),
    ("Kars", "Merkez", 40.601, 43.097, "Bazalt Platosu (ZB-ZC)", 520, "Kars / Kağızman Fay Zonu", 9.0),
    ("Kastamonu", "Merkez", 41.388, 33.782, "Fliş / Kireçtaşı (ZC)", 440, "KAF - Tosya/Ilgaz Segmenti", 20.0),
    ("Kayseri", "Melikgazi", 38.731, 35.478, "Erciyes Volkanik Tüf / Alüvyon (ZC)", 330, "Ecemiş Fay Zonu", 7.0),
    ("Kırıkkale", "Merkez", 39.846, 33.515, "Kızılırmak Alüvyonu / Kaya (ZC)", 360, "Kırıkkale-Erbaa Fayı", 8.0),
    ("Kırklareli", "Merkez", 41.733, 27.225, "Istranca Kristalin Masifi (ZB)", 680, "Trakya Masifi (Düşük Tektonik)", 3.0),
    ("Kırşehir", "Merkez", 39.142, 34.170, "Kristalin Temel / Çökel (ZC)", 430, "Seyfe / Orta Anadolu Fayı", 6.0),
    ("Kilis", "Merkez", 36.718, 37.121, "Kireçtaşı / Bazalt (ZC)", 480, "Ölü Deniz Fay Zonu Etkisi", 9.0),
    ("Kocaeli", "Gölcük", 40.716, 29.819, "Kıyı Alüvyonu / Delta (ZE)", 200, "KAF - İzmit Segmenti", 24.0),
    ("Konya", "Selçuklu", 37.871, 32.484, "Konya Kapalı Havza Çökeli (ZD)", 260, "Konya Fay Zonu", 5.0),
    ("Kütahya", "Merkez", 39.416, 29.983, "Havza Alüvyonu (ZD)", 255, "Simav / Kütahya Fay Zonu", 14.0),
    ("Malatya", "Yeşilyurt", 38.355, 38.309, "Malatya Havzası Alüvyonu (ZD)", 245, "Malatya-Ovacık / DAF", 11.0),
    ("Manisa", "Yunusemre", 38.619, 27.428, "Gediz Havzası Kalın Alüvyon (ZE)", 205, "Manisa / Gediz Graben Fayı", 18.0),
    ("Mardin", "Artuklu", 37.312, 40.735, "Mardin Eşiği Sağlam Kireçtaşı (ZB)", 660, "Arap Platformu (Düşük Tektonik)", 4.0),
    ("Mersin", "Yenişehir", 36.800, 34.633, "Kıyı Alüvyonu / Kireçtaşı (ZC)", 320, "Kıbrıs Yayı / Ecemiş", 10.0),
    ("Muğla", "Menteşe", 37.215, 28.363, "Karstik Kireçtaşı / Polye (ZC)", 460, "Muğla-Yatağan / Gökova Fayı", 22.0),
    ("Muş", "Merkez", 38.743, 41.506, "Muş Ovası Kalın Alüvyon (ZE)", 205, "Muş Bindirme / Varto Fayı", 14.0),
    ("Nevşehir", "Merkez", 38.624, 34.723, "Kapadokya Tüf / İgnimbrit (ZC)", 420, "Orta Anadolu / Derinkuyu", 5.0),
    ("Niğde", "Merkez", 37.966, 34.685, "Volkanik / Havza Çökeli (ZC)", 390, "Ecemiş Fay Zonu", 7.0),
    ("Ordu", "Altınordu", 40.983, 37.876, "Kıyı Alüvyonu / Volkanik (ZC)", 410, "Karadeniz Kıyı Kuşağı", 4.0),
    ("Osmaniye", "Merkez", 37.074, 36.247, "Havza Alüvyonu (ZD)", 240, "Doğu Anadolu / Karataş Fayı", 11.0),
    ("Rize", "Merkez", 41.020, 40.523, "Volkanik Kaya / Kıyı Dolgusu (ZC)", 480, "Doğu Karadeniz Kuşağı", 4.0),
    ("Sakarya", "Adapazarı", 40.773, 30.403, "Sıvılaşma Potansiyelli Alüvyon (ZE)", 185, "KAF - Sapanca/Akyazı", 24.0),
    ("Samsun", "Atakum", 41.292, 36.331, "Kıyı Alüvyonu / Kumtaşı (ZD)", 265, "KAF - Ladik/Havza Etkisi", 16.0),
    ("Siirt", "Merkez", 37.933, 41.950, "Kireçtaşı / Marn (ZC)", 440, "Bitlis-Zagros Bindirme Kuşağı", 9.0),
    ("Sinop", "Merkez", 42.023, 35.153, "Kumtaşı / Kireçtaşı (ZC)", 470, "Karadeniz Kuşağı (Düşük Tektonik)", 4.0),
    ("Sivas", "Merkez", 39.747, 37.017, "Kızılırmak Havzası / Jips (ZC)", 330, "Orta Anadolu / KAF Güney", 9.0),
    ("Şanlıurfa", "Haliliye", 37.159, 38.796, "Kireçtaşı Platosu (ZB-ZC)", 580, "Arap Platformu / Bozova Fayı", 5.0),
    ("Şırnak", "Merkez", 37.516, 42.461, "Cudi Kireçtaşı / Sıkışma Zonu (ZB)", 550, "Bitlis-Zagros Kenet Kuşağı", 9.0),
    ("Tekirdağ", "Süleymanpaşa", 40.983, 27.516, "Kıyı Çökelleri / Kumtaşı (ZD)", 255, "KAF - Ganos / Marmara", 24.0),
    ("Tokat", "Merkez", 40.316, 36.554, "Yeşilırmak Havza Alüvyonu (ZD)", 245, "KAF - Niksar/Erbaa Segmenti", 24.0),
    ("Trabzon", "Ortahisar", 41.001, 39.716, "Bazalt / Volkanik Kaya (ZB-ZC)", 530, "Doğu Karadeniz Kuşağı", 4.0),
    ("Tunceli", "Merkez", 39.107, 39.540, "Kireçtaşı / Vadide Çökel (ZC)", 450, "Ovacık / KAF-DAF Arası", 12.0),
    ("Uşak", "Merkez", 38.682, 29.408, "Havza Çökelleri / Metamorfik (ZC)", 340, "Simav / Gediz Etkisi", 14.0),
    ("Van", "İpekyolu", 38.494, 43.380, "Van Gölü Çökelleri / Alüvyon (ZD)", 230, "Van / Erciş Fay Zonu", 12.0),
    ("Yalova", "Merkez", 40.650, 29.266, "Kıyı Alüvyonu / Delta (ZE)", 195, "KAF - Çınarcık Segmenti", 24.0),
    ("Yozgat", "Merkez", 39.818, 34.814, "Granit / Kristalin Temel (ZB)", 620, "Orta Anadolu İç Kesim", 4.0),
    ("Zonguldak", "Merkez", 41.456, 31.798, "Karbonifer Kumtaşı / Kaya (ZB)", 590, "Batı Karadeniz Kuşağı", 5.0),
]


# ==========================================
# 3. HESAPLAMA VE USGS + AFAD CANLI VERİ ÇEKME
# ==========================================
def fetch_usgs(starttime, endtime, min_mag):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "csv",
        "starttime": starttime.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": endtime.strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude": MIN_LAT,
        "maxlatitude": MAX_LAT,
        "minlongitude": MIN_LON,
        "maxlongitude": MAX_LON,
        "minmagnitude": min_mag,
    }
    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if len(df) == 0:
        return pd.DataFrame(columns=["time", "latitude", "longitude", "depth", "mag", "place", "source"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["source"] = "USGS"
    return df[["time", "latitude", "longitude", "depth", "mag", "place", "source"]]


def fetch_afad(starttime_utc, endtime_utc, min_mag=1.2):
    url = "https://deprem.afad.gov.tr/apiv2/event/filter"
    params = {
        "start": starttime_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "end": endtime_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "minlat": MIN_LAT,
        "maxlat": MAX_LAT,
        "minlon": MIN_LON,
        "maxlon": MAX_LON,
        "minmag": min_mag,
        "format": "json",
    }
    headers = {"User-Agent": "Mozilla/5.0 (SeismicMonitor/1.0)"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame(columns=["time", "latitude", "longitude", "depth", "mag", "place", "source"])

        rows = []
        for item in data:
            t_utc = pd.to_datetime(item["date"], utc=True)
            rows.append({
                "time": t_utc,
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "depth": float(item["depth"]),
                "mag": float(item["magnitude"]),
                "place": str(item.get("location", "Türkiye")),
                "source": "AFAD"
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[UYARI] AFAD API hatası, USGS ile devam ediliyor: {e}")
        return pd.DataFrame(columns=["time", "latitude", "longitude", "depth", "mag", "place", "source"])


def merge_and_deduplicate(df_afad, df_usgs):
    if len(df_usgs) > 0 and "source" not in df_usgs.columns:
        df_usgs = df_usgs.copy()
        df_usgs["source"] = "USGS"

    if len(df_afad) == 0:
        return df_usgs
    if len(df_usgs) == 0:
        return df_afad

    merged = df_afad.copy()
    new_usgs_rows = []

    for _, u_row in df_usgs.iterrows():
        time_diffs = (merged["time"] - u_row["time"]).abs().dt.total_seconds()
        candidates = merged[time_diffs <= 20.0]

        is_duplicate = False
        if len(candidates) > 0:
            dists = haversine_km(
                u_row["latitude"], u_row["longitude"],
                candidates["latitude"].values, candidates["longitude"].values
            )
            if np.any(dists <= 25.0):
                is_duplicate = True

        if not is_duplicate:
            new_usgs_rows.append(u_row)

    if new_usgs_rows:
        merged = pd.concat([merged, pd.DataFrame(new_usgs_rows)], ignore_index=True)

    return merged.sort_values(by="time", ascending=False).reset_index(drop=True)


def haversine_km(lat1, lon1, lat_arr, lon_arr):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat_arr)
    dphi = np.radians(lat_arr - lat1)
    dlambda = np.radians(lon_arr - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def calculate_b_value(magnitudes):
    valid = magnitudes[magnitudes >= MC]
    if len(valid) < MIN_EVENTS_FOR_B:
        return None
    denom = np.mean(valid) - (MC - (DELTA_M / 2.0))
    return round(float(np.log10(np.e) / denom), 3) if denom > 0 else None


def get_soil_factor(vs30):
    return round(min(1.65, max(0.95, (760.0 / max(150.0, float(vs30))) ** 0.32)), 2)


def get_slip_rate(lat, lon):
    if 39.5 <= lat <= 41.3 and 26.0 <= lon <= 41.5:
        return 24.0, "Kuzey Anadolu Fay Zonu (KAF)"
    elif 35.5 <= lat <= 37.2 and 26.0 <= lon <= 35.5:
        return 30.0, "Helen-Kıbrıs Yayı"
    elif 37.0 <= lat <= 39.5 and 26.0 <= lon <= 30.5:
        return 18.0, "Batı Anadolu Graben Sistemi"
    elif 36.5 <= lat <= 39.8 and 35.5 <= lon <= 44.5:
        return 11.0, "Doğu Anadolu Fay Zonu (DAF)"
    return 4.0, "Orta Anadolu İkincil Tektonik"


def create_hexagon_vertices(c_lat, c_lon, r_deg):
    verts = []
    lon_scale = 1.0 / math.cos(math.radians(c_lat))
    for i in range(6):
        angle = math.radians(60 * i)
        v_lat = round(c_lat + r_deg * math.sin(angle), 4)
        v_lon = round(c_lon + r_deg * lon_scale * math.cos(angle), 4)
        verts.append([v_lat, v_lon])
    return verts


def run_live_pipeline():
    now_utc = datetime.now(timezone.utc)
    start_5y = now_utc - timedelta(days=5 * 365)
    start_50y = now_utc - timedelta(days=50 * 365)
    recent_cutoff = now_utc - timedelta(days=RECENT_DAYS)

    if HISTORICAL_CACHE["df_50y"] is None:
        print("[SİSTEM] 50 yıllık (M>=5) ve 5 yıllık (M>=2.6) tarihsel taban yükleniyor...")
        HISTORICAL_CACHE["df_50y"] = fetch_usgs(start_50y, now_utc, 5.0)
        mid = start_5y + timedelta(days=int(2.5 * 365))
        d1 = fetch_usgs(start_5y, mid, MC)
        d2 = fetch_usgs(mid, recent_cutoff, MC)
        HISTORICAL_CACHE["df_5y_base"] = pd.concat([d1, d2]).drop_duplicates().reset_index(drop=True)

    # Her 5 dakikada bir SON 30 GÜNÜ hem AFAD hem USGS'ten çekip tekilleştirerek birleştir
    df_usgs_live = fetch_usgs(recent_cutoff, now_utc, 2.0)
    df_afad_live = fetch_afad(recent_cutoff, now_utc, min_mag=1.2)
    df_recent_live = merge_and_deduplicate(df_afad_live, df_usgs_live)

    df_5y = pd.concat([
        HISTORICAL_CACHE["df_5y_base"][HISTORICAL_CACHE["df_5y_base"]["time"] >= start_5y],
        df_recent_live[df_recent_live["mag"] >= MC]
    ]).drop_duplicates(subset=["time", "latitude", "longitude"]).reset_index(drop=True)
    df_50y = HISTORICAL_CACHE["df_50y"]

    num_windows = (now_utc - start_5y).days // RECENT_DAYS

    # 3. Altıgen (Hexagonal) Tektonik Hücreleri Oluştur
    hex_list = []
    lat_step = HEX_RADIUS_DEG * 1.5
    lon_step = HEX_RADIUS_DEG * math.sqrt(3) * 1.25

    lat_vals = np.arange(MIN_LAT + 0.3, MAX_LAT - 0.2, lat_step)
    for row_idx, c_lat in enumerate(lat_vals):
        offset = (lon_step / 2.0) if (row_idx % 2 == 1) else 0.0
        lon_vals = np.arange(MIN_LON + 0.3 + offset, MAX_LON - 0.2, lon_step)
        for c_lon in lon_vals:
            c_lat_r, c_lon_r = round(float(c_lat), 3), round(float(c_lon), 3)
            dists_5y = haversine_km(c_lat_r, c_lon_r, df_5y["latitude"].values, df_5y["longitude"].values)
            local_5y = df_5y[dists_5y <= 34.0]
            if len(local_5y) < 4:
                continue

            dists_50y = haversine_km(c_lat_r, c_lon_r, df_50y["latitude"].values, df_50y["longitude"].values)
            local_50y = df_50y[dists_50y <= 34.0]

            slip_mm, fault_name = get_slip_rate(c_lat_r, c_lon_r)
            years_since = ((now_utc - local_50y["time"].max()).days / 365.25) if len(local_50y) > 0 else 50.0
            acc_slip = (slip_mm * years_since) / 1000.0
            gap_score = min(100.0, (acc_slip / 1.2) * 100.0)

            rec = local_5y[local_5y["time"] >= recent_cutoff]
            rec_cnt = len(rec)
            hist = local_5y[local_5y["time"] < recent_cutoff].copy()
            hist["win"] = (recent_cutoff - hist["time"]).dt.days // RECENT_DAYS
            wc = hist.groupby("win").size().reindex(range(num_windows - 1), fill_value=0)
            mean_30, std_30 = float(wc.mean()), float(wc.std())
            z_score = round((rec_cnt - mean_30) / std_30, 2) if std_30 > 0 else 0.0
            anom_score = min(100.0, max(0.0, (z_score / 3.5) * 100.0))

            b_val = calculate_b_value(local_5y[local_5y["time"] >= (now_utc - timedelta(days=365))]["mag"].values)
            stress_score = min(100.0, max(10.0, ((1.25 - b_val) / 0.55) * 100.0)) if b_val is not None else 45.0

            near_geo = any(haversine_km(c_lat_r, c_lon_r, np.array([g["lat"]]), np.array([g["lon"]]))[0] <= g["radius_km"] for g in GEOTHERMAL_ZONES)
            avg_depth = float(rec["depth"].mean()) if rec_cnt > 0 else float(local_5y["depth"].mean())
            is_swarm = bool(near_geo and avg_depth < 11.5 and (b_val is None or b_val > 0.95))
            geo_mult = 0.45 if is_swarm else 1.0

            score = round((0.40 * gap_score) + (0.35 * anom_score * geo_mult) + (0.25 * stress_score), 1)

            hex_list.append({
                "id": f"HEX-{c_lat_r}-{c_lon_r}",
                "lat": c_lat_r,
                "lon": c_lon_r,
                "vertices": create_hexagon_vertices(c_lat_r, c_lon_r, HEX_RADIUS_DEG),
                "fay": fault_name,
                "gps_mm": slip_mm,
                "son_m5_yil": round(years_since, 1),
                "biriken_atim_m": round(acc_slip, 2),
                "z_skoru": z_score,
                "b_degeri": b_val,
                "ort_derinlik": round(avg_depth, 1),
                "son_30g": rec_cnt,
                "aylik_ort": round(mean_30, 2),
                "jeotermal_yakin": near_geo,
                "firtina_filtresi": is_swarm,
                "skor": score,
            })

    # 4. 81 İl + İlçe Zemin Büyütmeli Risk Analizi
    city_list = []
    for il, ilce, lat, lon, zemin, vs30, fay, gps_mm in CITIES_DATA:
        local_50y = df_50y[haversine_km(lat, lon, df_50y["latitude"].values, df_50y["longitude"].values) <= SEARCH_RADIUS_KM]
        local_5y = df_5y[haversine_km(lat, lon, df_5y["latitude"].values, df_5y["longitude"].values) <= SEARCH_RADIUS_KM].copy()

        years_since = ((now_utc - local_50y["time"].max()).days / 365.25) if len(local_50y) > 0 else 50.0
        acc_slip = (gps_mm * years_since) / 1000.0
        gap_score = min(100.0, (acc_slip / 1.2) * 100.0)

        rec = local_5y[local_5y["time"] >= recent_cutoff]
        rec_cnt = len(rec)
        hist = local_5y[local_5y["time"] < recent_cutoff].copy()
        if len(hist) > 0:
            hist["win"] = (recent_cutoff - hist["time"]).dt.days // RECENT_DAYS
            wc = hist.groupby("win").size().reindex(range(num_windows - 1), fill_value=0)
            mean_30, std_30 = float(wc.mean()), float(wc.std())
            z_score = round((rec_cnt - mean_30) / std_30, 2) if std_30 > 0 else 0.0
        else:
            mean_30, z_score = 0.0, 0.0
        anom_score = min(100.0, max(0.0, (z_score / 3.5) * 100.0))

        b_val = calculate_b_value(local_5y[local_5y["time"] >= (now_utc - timedelta(days=365))]["mag"].values)
        stress_score = min(100.0, max(10.0, ((1.25 - b_val) / 0.55) * 100.0)) if b_val is not None else 45.0

        near_geo = any(haversine_km(lat, lon, np.array([g["lat"]]), np.array([g["lon"]]))[0] <= g["radius_km"] for g in GEOTHERMAL_ZONES)
        avg_depth = float(rec["depth"].mean()) if rec_cnt > 0 else 12.0
        is_swarm = bool(near_geo and avg_depth < 11.5 and (b_val is None or b_val > 0.95))
        geo_mult = 0.45 if is_swarm else 1.0

        tektonik_skor = round((0.45 * gap_score) + (0.30 * anom_score * geo_mult) + (0.25 * stress_score), 1)
        carpan = get_soil_factor(vs30)
        nihai_risk = min(100.0, round(tektonik_skor * carpan, 1))

        city_list.append({
            "il": il, "ilce": ilce, "lat": lat, "lon": lon,
            "zemin": zemin, "vs30": vs30, "carpan": carpan,
            "fay": fay, "gps_mm": gps_mm,
            "son_m5_yil": round(years_since, 1),
            "biriken_atim_m": round(acc_slip, 2),
            "z_skoru": z_score, "b_degeri": b_val,
            "son_30g": rec_cnt, "aylik_ort": round(mean_30, 2),
            "jeotermal_yakin": near_geo, "firtina_filtresi": is_swarm,
            "tektonik_skor": tektonik_skor, "nihai_risk": nihai_risk
        })

    # 5. Son 7 Günün Canlı Depremleri (AFAD + USGS Birleşik Akış)
    last_7d = df_recent_live[df_recent_live["time"] >= (now_utc - timedelta(days=7))].sort_values(by="time", ascending=False).head(150)
    quakes_list = []
    for _, q in last_7d.iterrows():
        quakes_list.append({
            "time_tr": (q["time"] + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M"),
            "lat": round(float(q["latitude"]), 3),
            "lon": round(float(q["longitude"]), 3),
            "depth": round(float(q["depth"]), 1),
            "mag": round(float(q["mag"]), 1),
            "place": f"[{q.get('source', 'AFAD')}] {q['place']}",
        })

    LIVE_STATE["updated_at"] = (now_utc + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M:%S (TSİ)")
    LIVE_STATE["hexagons"] = hex_list
    LIVE_STATE["cities"] = sorted(city_list, key=lambda x: x["nihai_risk"], reverse=True)
    LIVE_STATE["geothermal"] = GEOTHERMAL_ZONES
    LIVE_STATE["recent_quakes"] = quakes_list
    print(f"[CANLI GÜNCELLEME] {LIVE_STATE['updated_at']} -> {len(hex_list)} Altıgen, {len(city_list)} Şehir, {len(quakes_list)} Canlı Deprem (AFAD+USGS) güncellendi.")


async def periodic_updater():
    while True:
        try:
            await asyncio.to_thread(run_live_pipeline)
        except Exception as e:
            print(f"[HATA] Canlı güncelleme sırasında hata oluştu: {e}")
        await asyncio.sleep(REFRESH_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(periodic_updater())
    yield
    task.cancel()


app = FastAPI(title="Türkiye Canlı Sismik Anomali ve Zemin Risk Platformu", lifespan=lifespan)


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/api/live-data")
def get_live_data():
    return JSONResponse(content=LIVE_STATE)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)