"""
Embedded knowledge base (fallback dataset).
Used when MySQL is not configured; also the seed source for the DB import.
Covers: spots, coordinates, fuel estimation, recommendations.
"""

from typing import Dict, List, Tuple

# Tourist spots
# Format: { spot_id: { name, location, description, category, lat, lng } }

TOURIST_SPOTS: Dict[str, dict] = {
    "calle_crisologo": {
        "name": "Calle Crisologo",
        "location": "Vigan City",
        "description": (
            "A UNESCO World Heritage cobblestone street lined with Spanish colonial "
            "houses and kalesas (horse-drawn carriages). The most iconic street in Vigan."
        ),
        "category": ["heritage", "history", "culture"],
        "lat": 17.5742,
        "lng": 120.3874,
        "highlights": ["kalesa rides", "antique shops", "Spanish architecture"],
        "best_time": "Early morning or evening to avoid heat",
    },
    "vigan_cathedral": {
        "name": "Vigan Cathedral (St. Paul Metropolitan Cathedral)",
        "location": "Vigan City",
        "description": (
            "A baroque-style Roman Catholic cathedral built in the 16th century. "
            "One of the oldest churches in the Philippines."
        ),
        "category": ["heritage", "religion", "history"],
        "lat": 17.5748,
        "lng": 120.3882,
        "highlights": ["baroque architecture", "bell tower", "historical artifacts"],
        "best_time": "Morning for mass or sightseeing",
    },
    "plaza_salcedo": {
        "name": "Plaza Salcedo",
        "location": "Vigan City",
        "description": (
            "The main plaza of Vigan featuring a dancing fountain show at night. "
            "A great spot for relaxing and experiencing local life."
        ),
        "category": ["landmark", "leisure", "culture"],
        "lat": 17.5750,
        "lng": 120.3879,
        "highlights": ["dancing fountain", "heritage buildings", "local food stalls"],
        "best_time": "Evening for the fountain show (7 PM - 9 PM)",
    },
    "syquia_mansion": {
        "name": "Syquia Mansion",
        "location": "Vigan City",
        "description": (
            "Ancestral home of former Philippine President Elpidio Quirino. "
            "Now a museum showcasing 19th-century Filipino-Spanish lifestyle."
        ),
        "category": ["museum", "heritage", "history"],
        "lat": 17.5739,
        "lng": 120.3870,
        "highlights": ["presidential artifacts", "antique furniture", "heritage architecture"],
        "best_time": "Daytime (open 8 AM - 5 PM)",
    },
    "burnay_pottery": {
        "name": "Burnay Pottery",
        "location": "Vigan City",
        "description": (
            "Traditional Ilocano pottery workshops where you can watch and try "
            "making burnay pots — a centuries-old craft unique to Vigan."
        ),
        "category": ["culture", "crafts", "experience"],
        "lat": 17.5720,
        "lng": 120.3850,
        "highlights": ["pottery making demo", "buy local crafts", "kiln viewing"],
        "best_time": "Morning when artisans are most active",
    },
    "unp": {
        "name": "University of Northern Philippines (UNP)",
        "location": "Vigan City",
        "description": (
            "One of the oldest state universities in the Philippines, founded in 1906. "
            "Located in Tamag, Vigan City."
        ),
        "category": ["landmark", "education"],
        "lat": 17.5625,
        "lng": 120.3923,
        "highlights": ["historical campus", "museum", "academic landmark"],
        "best_time": "Weekdays during school hours",
    },
    "bantay_church": {
        "name": "Bantay Church & Bell Tower",
        "location": "Bantay, Ilocos Sur",
        "description": (
            "A 16th-century Spanish colonial church with a separate bell tower used "
            "as a watchtower during the Spanish era. Offers panoramic views of Vigan."
        ),
        "category": ["heritage", "religion", "history"],
        "lat": 17.5893,
        "lng": 120.3864,
        "highlights": ["panoramic view", "bell tower climb", "centuries-old church"],
        "best_time": "Late afternoon for sunset views",
    },
    "baluarte": {
        "name": "Baluarte Zoo",
        "location": "Vigan City",
        "description": (
            "A free zoo owned by Governor Chavit Singson featuring exotic animals "
            "including white tigers, crocodiles, and various birds."
        ),
        "category": ["nature", "family", "leisure"],
        "lat": 17.5558,
        "lng": 120.3869,
        "highlights": ["white tigers", "free admission", "exotic animals"],
        "best_time": "Morning (animals are more active)",
    },
    "mindoro_beach": {
        "name": "Mindoro Beach",
        "location": "Santa, Ilocos Sur",
        "description": (
            "A serene black sand beach in the municipality of Santa, "
            "ideal for swimming, relaxing, and watching sunsets."
        ),
        "category": ["beach", "nature", "leisure"],
        "lat": 17.4923,
        "lng": 120.4134,
        "highlights": ["black sand beach", "swimming", "sunset views"],
        "best_time": "Late afternoon for sunset",
    },
    "pinsal_falls": {
        "name": "Pinsal Falls",
        "location": "San Esteban, Ilocos Sur",
        "description": (
            "A beautiful multi-tier waterfall surrounded by lush greenery in San Esteban. "
            "A hidden gem for nature lovers and adventure seekers."
        ),
        "category": ["nature", "adventure", "eco-tourism"],
        "lat": 17.3800,
        "lng": 120.4500,
        "highlights": ["waterfall swimming", "nature trek", "picnic area"],
        "best_time": "Rainy season (July-September) for full water flow",
    },
    "sta_maria_church": {
        "name": "Santa Maria Church",
        "location": "Santa Maria, Ilocos Sur",
        "description": (
            "A UNESCO World Heritage Site — a hilltop Baroque church built in 1765, "
            "one of the four Philippine Baroque churches listed by UNESCO."
        ),
        "category": ["heritage", "religion", "history", "UNESCO"],
        "lat": 17.6258,
        "lng": 120.4897,
        "highlights": ["UNESCO heritage", "hilltop views", "baroque architecture", "87 stone steps"],
        "best_time": "Morning for cooler weather to climb the steps",
    },
    "vigan_plaza_burgos": {
        "name": "Plaza Burgos",
        "location": "Vigan City",
        "description": (
            "A vibrant plaza and market area in Vigan known for local snacks, "
            "longanisa (Ilocano sausage), and street food."
        ),
        "category": ["food", "culture", "leisure"],
        "lat": 17.5744,
        "lng": 120.3891,
        "highlights": ["Vigan longanisa", "bagnet", "local delicacies", "night market"],
        "best_time": "Early morning for fresh market or evening for street food",
    },
}

# Car fuel consumption (L/100km), city/highway combined

CAR_TYPES: Dict[str, dict] = {
    "motorcycle": {
        "label": "Motorcycle / Motorbike",
        "consumption_per_100km": 3.0,
        "aliases": ["motorcycle", "motorbike", "motor", "bike", "moto"],
    },
    "sedan": {
        "label": "Sedan (e.g. Toyota Vios, Honda City)",
        "consumption_per_100km": 8.5,
        "aliases": ["sedan", "car", "vios", "civic", "city", "altis", "camry"],
    },
    "suv": {
        "label": "SUV (e.g. Ford Everest, Toyota Fortuner)",
        "consumption_per_100km": 12.0,
        "aliases": ["suv", "fortuner", "everest", "prado", "mux", "mu-x", "crv", "cr-v"],
    },
    "van": {
        "label": "Van (e.g. Toyota Hiace, Foton)",
        "consumption_per_100km": 13.5,
        "aliases": ["van", "hiace", "foton", "urvan", "grandia"],
    },
    "pickup": {
        "label": "Pickup Truck (e.g. Toyota Hilux, Ford Ranger)",
        "consumption_per_100km": 11.0,
        "aliases": ["pickup", "hilux", "ranger", "truck", "l300"],
    },
    "multicab": {
        "label": "Multicab / Jeepney",
        "consumption_per_100km": 9.0,
        "aliases": ["multicab", "jeepney", "jeep", "multicab"],
    },
    "electric": {
        "label": "Electric Vehicle",
        "consumption_per_100km": 0.0,  # kWh based, handle separately
        "aliases": ["electric", "ev", "tesla", "atto", "byd"],
    },
}

# Fuel prices (PHP per liter)
FUEL_PRICES: Dict[str, float] = {
    "gasoline": 62.0,   # PHP per liter (approximate 2024 avg)
    "diesel": 58.0,
    "premium": 68.0,
}

# Distance matrix (km) — approximate road distances
DISTANCES_KM: Dict[Tuple[str, str], float] = {
    ("calle_crisologo", "vigan_cathedral"): 0.2,
    ("calle_crisologo", "plaza_salcedo"): 0.3,
    ("calle_crisologo", "syquia_mansion"): 0.4,
    ("calle_crisologo", "burnay_pottery"): 0.7,
    ("calle_crisologo", "unp"): 2.5,
    ("calle_crisologo", "bantay_church"): 2.8,
    ("calle_crisologo", "baluarte"): 3.2,
    ("calle_crisologo", "mindoro_beach"): 12.0,
    ("calle_crisologo", "pinsal_falls"): 28.0,
    ("calle_crisologo", "sta_maria_church"): 55.0,
    ("calle_crisologo", "vigan_plaza_burgos"): 0.5,
    ("unp", "calle_crisologo"): 2.5,
    ("unp", "bantay_church"): 3.5,
    ("unp", "baluarte"): 2.1,
    ("unp", "vigan_cathedral"): 2.3,
    ("bantay_church", "vigan_cathedral"): 2.6,
    ("bantay_church", "baluarte"): 4.0,
    ("baluarte", "mindoro_beach"): 13.0,
    ("mindoro_beach", "pinsal_falls"): 20.0,
    ("vigan_cathedral", "plaza_salcedo"): 0.2,
    ("plaza_salcedo", "syquia_mansion"): 0.3,
    ("syquia_mansion", "burnay_pottery"): 0.5,
}

# Recommendations by category
CATEGORY_RECOMMENDATIONS: Dict[str, List[str]] = {
    "heritage": ["calle_crisologo", "vigan_cathedral", "bantay_church", "sta_maria_church", "syquia_mansion"],
    "nature": ["pinsal_falls", "mindoro_beach", "baluarte"],
    "beach": ["mindoro_beach"],
    "food": ["vigan_plaza_burgos", "burnay_pottery", "calle_crisologo"],
    "family": ["baluarte", "plaza_salcedo", "calle_crisologo", "burnay_pottery"],
    "adventure": ["pinsal_falls"],
    "history": ["calle_crisologo", "vigan_cathedral", "sta_maria_church", "syquia_mansion", "bantay_church"],
    "culture": ["calle_crisologo", "burnay_pottery", "vigan_plaza_burgos"],
    "must_see": ["calle_crisologo", "bantay_church", "sta_maria_church", "baluarte", "plaza_salcedo"],
}
