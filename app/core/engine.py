"""
Smartour AI Engine
------------------
Rule-based + TF-IDF intent assistant for Ilocos Sur tourism, fully offline.
"""

import re
import math
from difflib import get_close_matches
from typing import List, Optional, Tuple

from app.knowledge.knowledge_base import (
    TOURIST_SPOTS,
    CAR_TYPES,
    FUEL_PRICES,
    DISTANCES_KM,
    CATEGORY_RECOMMENDATIONS,
    MUNICIPAL_DISTANCES,
)
from app.training.trainable_model import get_intent_model


# Helpers

def _normalize(text: str) -> str:
    return text.lower().strip()


def _fuzzy_match_spot(query: str) -> Optional[str]:
    query = _normalize(query)

    if query in TOURIST_SPOTS:
        return query

    name_map = {_normalize(v["name"]): k for k, v in TOURIST_SPOTS.items()}
    location_map = {_normalize(v["location"]): k for k, v in TOURIST_SPOTS.items()}

    for name, spot_id in name_map.items():
        if name in query or query in name:
            return spot_id
    for loc, spot_id in location_map.items():
        if loc in query:
            return spot_id

    # Spot id used directly as a word (e.g. "unp")
    query_tokens = set(re.findall(r"[a-z0-9]+", query))
    for sid in TOURIST_SPOTS:
        if _normalize(sid) in query_tokens:
            return sid

    # Token overlap — most reliable for long phrases with typos
    best_sid, best_score = None, 0
    for name, spot_id in name_map.items():
        name_tokens = set(re.findall(r"[a-z0-9]+", name))
        overlap = len(query_tokens & name_tokens)
        if overlap >= 2 and overlap > best_score:
            best_sid, best_score = spot_id, overlap
    if best_sid:
        return best_sid

    # Fuzzy fallback (last resort — helps short typo'd names like "balwarte")
    all_names = list(name_map.keys())
    matches = get_close_matches(query, all_names, n=1, cutoff=0.5)
    if matches:
        return name_map[matches[0]]

    return None


def _find_spot_reference(text: str) -> Optional[str]:
    """Best-effort: find a spot the user mentioned, even in a longer phrase."""
    spot_id = _fuzzy_match_spot(text)
    if spot_id:
        return spot_id

    text_l = _normalize(text)
    # Distinctive first word of each spot name, word-boundary matched
    for sid, sdata in TOURIST_SPOTS.items():
        first_word = _normalize(sdata["name"]).split()[0]
        if len(first_word) >= 3 and re.search(rf"\b{re.escape(first_word)}\b", text_l):
            return sid
    for sid, sdata in TOURIST_SPOTS.items():
        if _normalize(sdata["location"]) in text_l:
            return sid
    return None


def _spots_in_order(text: str) -> List[str]:
    """Return spot IDs mentioned in a message, in order of appearance."""
    text_l = _normalize(text)
    hits = []  # (position, exact_name_match, spot_id)
    for sid, sdata in TOURIST_SPOTS.items():
        name = _normalize(sdata["name"])
        if name in text_l:
            hits.append((text_l.find(name), True, sid))
        elif sid in text_l:
            hits.append((text_l.find(sid), True, sid))
        else:
            # Partial name: match the distinctive first word, word-boundary
            first_word = name.split()[0]
            if len(first_word) >= 3:
                m = re.search(rf"\b{re.escape(first_word)}\b", text_l)
                if m:
                    hits.append((m.start(), False, sid))
    # Exact matches win over partial ones at the same position
    hits.sort(key=lambda h: (h[0], not h[1]))
    result = []
    for pos, exact, sid in hits:
        if result and pos == result[-1][0] and not exact:
            continue  # drop partial duplicates (e.g. two "plaza" spots)
        result.append((pos, sid))
    return [sid for _, sid in result]


def _detect_car_type(text: str) -> Optional[str]:
    text = _normalize(text)
    for car_id, car_data in CAR_TYPES.items():
        for alias in car_data["aliases"]:
            if alias in text:
                return car_id
    return None


# Words that don't carry meaning when a user just answers the vehicle prompt
_FILLER_WORDS = {
    "a", "an", "the", "and", "with", "using", "use", "for", "to", "me",
    "i", "have", "my", "is", "it", "s", "what", "about", "just",
    "please", "pls", "po", "ako", "ng", "na", "lang", "yes", "ok", "okay",
}


def _is_bare_car_reply(message: str) -> bool:
    """True when the message is just a vehicle reply (e.g. "motorcycle", "a sedan")."""
    if not _detect_car_type(message):
        return False
    text = _normalize(message)
    for car_data in CAR_TYPES.values():
        for alias in car_data["aliases"]:
            # Word-boundary so "motor" doesn't strip inside "motorcycle"
            text = re.sub(rf"\b{re.escape(alias)}\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    leftovers = {t for t in text.split() if t not in _FILLER_WORDS}
    return not leftovers


def _get_distance(from_id: str, to_id: str) -> Optional[float]:
    key1 = (from_id, to_id)
    key2 = (to_id, from_id)
    return DISTANCES_KM.get(key1) or DISTANCES_KM.get(key2)


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Fallback: straight-line distance using Haversine formula (km)."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * 1.3  # 1.3 road factor


def _estimate_distance(from_id: str, to_id: str) -> float:
    d = _get_distance(from_id, to_id)
    if d:
        return d
    s1 = TOURIST_SPOTS[from_id]
    s2 = TOURIST_SPOTS[to_id]
    return round(_haversine_km(s1["lat"], s1["lng"], s2["lat"], s2["lng"]), 1)


def _fuel_cost(distance_km: float, car_id: str, fuel_type: str = "gasoline") -> dict:
    car = CAR_TYPES[car_id]
    price_per_liter = FUEL_PRICES.get(fuel_type, FUEL_PRICES["gasoline"])

    liters_needed = (distance_km / 100) * car["consumption_per_100km"]
    cost = liters_needed * price_per_liter

    return {
        "distance_km": distance_km,
        "car": car["label"],
        "fuel_type": fuel_type.capitalize(),
        "liters_needed": round(liters_needed, 2),
        "price_per_liter": price_per_liter,
        "estimated_cost_php": round(cost, 2),
    }


# Intent detection

INTENT_PATTERNS = {
    "fuel_cost": [
        r"fuel", r"gas", r"gasoline", r"diesel", r"cost", r"how much.*travel",
        r"petrol", r"liters?", r"consume", r"consumption", r"from .+ to", r"magkano",
    ],
    "recommend": [
        r"best place", r"suggest", r"recommend", r"where.*go", r"what.*visit",
        r"top spot", r"must.?see", r"saan.*punta", r"where should", r"what to see",
        r"tourist spot", r"places? to visit", r"what are.*spot",
    ],
    "describe": [
        r"tell me about", r"what is", r"describe", r"info.*about", r"about",
        r"details?", r"what.*know about",
    ],
    "distance": [
        r"how far", r"distance", r"km", r"kilometer", r"how many km",
        r"gaano kalayo",
    ],
    "greeting": [
        r"^hi\b", r"^hello\b", r"^hey\b", r"kumusta", r"good morning",
        r"good afternoon", r"good evening", r"musta",
    ],
    "help": [
        r"help", r"what can you do", r"how.*use", r"commands?", r"features?",
    ],
}


def _detect_intent(text: str) -> str:
    text = _normalize(text)
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent
    return "unknown"


def _detect_fuel_type(text: str) -> str:
    text = _normalize(text)
    if "diesel" in text:
        return "diesel"
    if "premium" in text:
        return "premium"
    return "gasoline"


def _wants_fuel(text: str) -> bool:
    """True when the message asks for a fuel estimate (or names a vehicle)."""
    text = _normalize(text)
    if any(kw in text for kw in (
        "fuel", "gas", "gasoline", "diesel", "petrol",
        "consumption", "liters", "liter", "magkano",
    )):
        return True
    return _detect_car_type(text) is not None


def _fuel_estimate_block(
    from_id: str, to_id: str, message: str, context: Optional[dict] = None
) -> str:
    car_id = _detect_car_type(message) or (context or {}).get("car_id")
    if not car_id:
        return (
            "Want a fuel estimate too? Tell me your vehicle type:\n"
            "- Motorcycle, Sedan, SUV, Van, Pickup, Multicab\n\n"
            "Example: _\"... using a sedan\"_"
        )
    fuel_type = _detect_fuel_type(message)
    distance = _estimate_distance(from_id, to_id)
    return _build_fuel_response(_fuel_cost(distance, car_id, fuel_type))


# ─────────────────────────────────────────────────────────────────────────────
# Municipality proximity helpers
# ─────────────────────────────────────────────────────────────────────────────

# Default reference municipality when user doesn't specify one
DEFAULT_MUNICIPALITY = "Vigan City"

# Max road distance (km) to consider "nearby" for recommendations
# Anything beyond this shows in the "further away" section
NEARBY_THRESHOLD_KM = 15.0


def _get_municipality(spot_id: str) -> str:
    """Return the municipality name for a spot."""
    return TOURIST_SPOTS.get(spot_id, {}).get("location", "")


def _get_municipal_distance(muni_a: str, muni_b: str) -> float:
    """Return road distance between two municipalities in km.
    Returns 0 if same municipality, inf if not found."""
    if muni_a == muni_b:
        return 0.0
    key1 = (muni_a, muni_b)
    key2 = (muni_b, muni_a)
    d = MUNICIPAL_DISTANCES.get(key1) or MUNICIPAL_DISTANCES.get(key2)
    return d if d is not None else float("inf")


def _detect_reference_municipality(text: str, context: Optional[dict] = None) -> Optional[str]:
    """Detect the reference municipality from the user's message.
    Checks: 1) mentioned spot's municipality, 2) direct municipality name, 3) context."""
    text_l = _normalize(text)

    # 1) Spot mentioned in the message
    spot_id = _fuzzy_match_spot(text)
    if spot_id:
        return _get_municipality(spot_id)

    # 2) Direct municipality name match
    all_municipalities = {_get_municipality(sid) for sid in TOURIST_SPOTS}
    for muni in all_municipalities:
        if muni and muni.lower() in text_l:
            return muni

    # 3) Context: last spot discussed
    if context:
        ctx_spot = context.get("spot_id")
        if ctx_spot and ctx_spot in TOURIST_SPOTS:
            return _get_municipality(ctx_spot)

    return None


def _sort_spots_by_proximity(spot_ids: List[str], reference_municipality: str) -> List[str]:
    """Sort spots by distance from the reference municipality.
    Spots in the same municipality come first, then by road distance."""
    def _sort_key(sid: str) -> Tuple[float, int]:
        muni = _get_municipality(sid)
        dist = _get_municipal_distance(reference_municipality, muni)
        # Same municipality = distance 0, then by original index to preserve category order
        return (dist, spot_ids.index(sid))
    return sorted(spot_ids, key=_sort_key)


def _filter_nearby_spots(spot_ids: List[str], reference_municipality: str,
                          threshold_km: float = NEARBY_THRESHOLD_KM) -> Tuple[List[str], List[str]]:
    """Split spots into nearby and far groups based on municipal distance.
    Returns (nearby_spots, far_spots)."""
    nearby, far = [], []
    for sid in spot_ids:
        muni = _get_municipality(sid)
        dist = _get_municipal_distance(reference_municipality, muni)
        if dist <= threshold_km:
            nearby.append(sid)
        else:
            far.append(sid)
    return nearby, far


def _detect_category(text: str) -> Optional[str]:
    text = _normalize(text)
    keyword_map = {
        "heritage": ["heritage", "history", "historical", "old", "colonial", "spanish"],
        "nature": ["nature", "waterfall", "falls", "green", "eco", "outdoor"],
        "beach": ["beach", "swim", "sea", "ocean", "sand"],
        "food": ["food", "eat", "restaurant", "delicacy", "longanisa", "bagnet", "snack"],
        "family": ["family", "kids", "children", "fun", "zoo", "animal"],
        "adventure": ["adventure", "extreme", "trek", "hike", "hiking"],
        "must_see": ["best", "must see", "top", "famous", "popular", "must visit"],
    }
    for cat, keywords in keyword_map.items():
        for kw in keywords:
            if " " in kw:
                if kw in text:
                    return cat
            elif re.search(rf"\b{re.escape(kw)}\b", text):
                # Word-boundary so "eco" doesn't fire inside "recommend"
                return cat
    return None


# Response builders

# Directions ("how do I get to X?")

# Phrases that ask HOW TO GET somewhere (vs. just "tell me about X")
_DIRECTIONS_PATTERNS = [
    r"how (do i|can i|to|do you) (get|go|reach)\b",
    r"\bdirections?\s+(to|for)\b",
    r"\bway (to|of getting)\b",
    r"paano (\w+ )?(pumunta|makarating|makapunta|makakarating|magpunta)\b",
    r"\bpapunta\b",
]

# City-center reference: Calle Crisologo (distance table covers all spots from it)
CITY_CENTER_SPOT = "calle_crisologo"


def _is_directions_query(text: str) -> bool:
    text = _normalize(text)
    return any(re.search(pattern, text) for pattern in _DIRECTIONS_PATTERNS)


def _nearest_spot(spot_id: str) -> Optional[Tuple[str, float]]:
    best_id, best_km = None, None
    for other_id in TOURIST_SPOTS:
        if other_id == spot_id:
            continue
        km = _estimate_distance(spot_id, other_id)
        if best_km is None or km < best_km:
            best_id, best_km = other_id, km
    return (best_id, best_km) if best_id else None


def _build_directions(spot_id: str) -> str:
    s = TOURIST_SPOTS[spot_id]
    lines = [f"**Getting to {s['name']}**", f"Location: {s['location']}"]

    if spot_id == CITY_CENTER_SPOT:
        lines.append("It's right in the heart of Vigan (Calle Crisologo).")
    else:
        km = _estimate_distance(CITY_CENTER_SPOT, spot_id)
        lines.append(f"Distance from Vigan city center (Calle Crisologo): **{km} km**")

    nearest = _nearest_spot(spot_id)
    if nearest:
        n_id, n_km = nearest
        lines.append(f"Nearest landmark: **{TOURIST_SPOTS[n_id]['name']}** (~{n_km} km away)")

    if s.get("best_time"):
        lines.append(f"Best time to visit: {s['best_time']}")
    return "\n".join(lines)


def _build_spot_card(spot_id: str) -> str:
    s = TOURIST_SPOTS[spot_id]
    highlights = ", ".join(s.get("highlights", []))
    return (
        f"**{s['name']}**\n"
        f"Location: {s['location']}\n"
        f"{s['description']}\n"
        f"Highlights: {highlights}\n"
        f"Best time to visit: {s.get('best_time', 'Anytime')}"
    )


def _build_fuel_response(info: dict) -> str:
    if "energy_needed" in info:
        return (
            f"**Fuel Estimate (Electric Vehicle)**\n"
            f"Distance: {info['distance_km']} km\n"
            f"Energy needed: {info['energy_needed']}\n"
            f"Cost: {info['estimated_cost']}"
        )
    return (
        f"**Fuel Estimate**\n"
        f"Vehicle: {info['car']}\n"
        f"Distance: {info['distance_km']} km\n"
        f"Fuel type: {info['fuel_type']}\n"
        f"Fuel needed: {info['liters_needed']} liters\n"
        f"Price per liter: ₱{info['price_per_liter']:.2f}\n"
        f"Estimated cost: **₱{info['estimated_cost_php']:.2f}**"
    )


def _truncate(text: str, limit: int = 100) -> str:
    """Truncate at a word boundary, adding '…' only when actually cut."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _build_recommendation_list(spot_ids: List[str], title: str) -> str:
    lines = [f"**{title}**\n"]
    for i, sid in enumerate(spot_ids[:6], 1):
        s = TOURIST_SPOTS[sid]
        lines.append(f"{i}. **{s['name']}** — {s['location']}\n   _{_truncate(s['description'])}_")
    lines.append("\nType the name of any spot and I'll give you more details!")
    return "\n".join(lines)


def _build_proximity_recommendation(
    spot_ids: List[str], title: str, reference_municipality: str
) -> str:
    """Build a recommendation list that prioritizes nearby spots."""
    sorted_ids = _sort_spots_by_proximity(spot_ids, reference_municipality)
    nearby, _ = _filter_nearby_spots(sorted_ids, reference_municipality)

    lines = [f"**{title}**\n"]

    if nearby:
        for i, sid in enumerate(nearby[:6], 1):
            s = TOURIST_SPOTS[sid]
            dist = _get_municipal_distance(
                reference_municipality, _get_municipality(sid)
            )
            dist_label = "" if dist == 0 else f" (~{dist:.0f} km)"
            lines.append(
                f"{i}. **{s['name']}** — {s['location']}{dist_label}\n"
                f"   _{_truncate(s['description'])}_"
            )
    else:
        lines.append("No nearby spots found for this category.")

    lines.append("\nType the name of any spot and I'll give you more details!")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Clarifying questions (used when the bot is unsure)
# ─────────────────────────────────────────────────────────────────────────────

# Nudges keyed by the intent the model almost predicted (below the threshold)
_CLARIFY_HINTS = {
    "recommend": (
        "It sounds like you're looking for **places to visit**.\n\n"
        "Tell me what you're interested in — heritage, nature, beaches, food, "
        "family, or adventure — and I'll recommend spots.\n\n"
        "Try: _\"recommend heritage spots\"_ or _\"best beaches in Ilocos Sur\"_"
    ),
    "describe": (
        "It sounds like you're asking about a **specific place**.\n\n"
        "Just name it! Known spots: Calle Crisologo, Vigan Cathedral, Bantay "
        "Church, Baluarte, Pinsal Falls, Santa Maria Church, Mindoro Beach...\n\n"
        "Try: _\"tell me about Baluarte\"_"
    ),
    "fuel_cost": (
        "It sounds like you want a **fuel estimate**.\n\n"
        "Tell me your route and vehicle: _\"from [spot A] to [spot B] using "
        "a [car type]\"_.\n\n"
        "Example: _\"from Calle Crisologo to Bantay Church using a sedan\"_"
    ),
    "distance": (
        "It sounds like you're asking about a **distance**.\n\n"
        "Tell me two places: _\"how far is Bantay Church from Vigan "
        "Cathedral?\"_"
    ),
    "help": (
        "Not sure what you meant there — but I can help with "
        "**recommendations**, **spot info**, **fuel costs**, and **distances**.\n\n"
        "Type **help** to see the full list of things I can do!"
    ),
}


# Conversation context

def _extract_context(history: List[dict]) -> dict:
    """Remember route/car/spot from recent history so follow-ups can reuse it."""
    ctx: dict = {
        "from_id": None, "to_id": None, "car_id": None,
        "spot_id": None, "last_topic": None,
    }
    if not history:
        return ctx
    for msg in reversed(history[:-1][-10:]):
        content = msg.get("content") or ""
        role = msg.get("role")
        if role == "assistant":
            if not ctx["last_topic"]:
                ctx["last_topic"] = _topic_of_reply(content)
            # Also extract routes from assistant replies (e.g. "Distance from X to Y")
            if not ctx["from_id"]:
                route = _route_from_assistant_reply(content)
                if route:
                    ctx["from_id"], ctx["to_id"] = route
            continue
        route = _route_from_text(content)
        if route and not ctx["from_id"]:
            ctx["from_id"], ctx["to_id"] = route
        spots = _spots_in_order(content)
        if spots and not ctx["spot_id"]:
            ctx["spot_id"] = spots[-1]
        if not ctx["car_id"]:
            ctx["car_id"] = _detect_car_type(content)
        if ctx["from_id"] and ctx["car_id"] and ctx["last_topic"]:
            break
    return ctx


def _topic_of_reply(content: str) -> Optional[str]:
    if "Fuel Estimate" in content or "Trip:" in content or "vehicle type" in content:
        return "fuel_cost"
    if "Distance Estimate" in content or "Distance from" in content or "to where" in content:
        return "distance"
    if "Getting to" in content or "Location:" in content:
        return "describe"
    if "Must-See" in content or "**Top " in content or "**Recommended" in content:
        return "recommend"
    return None


def _route_from_assistant_reply(content: str) -> Optional[Tuple[str, str]]:
    """Extract route from assistant replies like 'Distance from X to Y' or 'Trip: X → Y'."""
    # Pattern: "Distance from **X** to **Y**"
    m = re.search(
        r"Distance from\s+\*\*(.+?)\*\*\s+to\s+\*\*(.+?)\*\*",
        content, re.IGNORECASE
    )
    if m:
        f = _fuzzy_match_spot(m.group(1).strip())
        t = _fuzzy_match_spot(m.group(2).strip())
        if f and t:
            return f, t
    # Pattern: "Trip: X → Y" or "Trip: X -> Y"
    m = re.search(
        r"Trip:\s+(.+?)\s+[→->]+\s+(.+?)(?:\n|$)",
        content, re.IGNORECASE
    )
    if m:
        f = _fuzzy_match_spot(m.group(1).strip())
        t = _fuzzy_match_spot(m.group(2).strip())
        if f and t:
            return f, t
    # Pattern: "From: **X**" and "To: **Y**" on separate lines
    from_m = re.search(r"From:\s+\*\*(.+?)\*\*", content, re.IGNORECASE)
    to_m = re.search(r"To:\s+\*\*(.+?)\*\*", content, re.IGNORECASE)
    if from_m and to_m:
        f = _fuzzy_match_spot(from_m.group(1).strip())
        t = _fuzzy_match_spot(to_m.group(1).strip())
        if f and t:
            return f, t
    return None


def _route_from_text(text: str) -> Optional[Tuple[str, str]]:
    """Extract the (from, to) route, handling "from X to Y" and "to X from Y"."""
    m = re.search(
        r"from\s+(.+?)\s+to\s+(.+?)(?:\s+using|\s+with|\s+i have|[,\.\?]|$)",
        text, re.IGNORECASE
    )
    if m:
        f = _fuzzy_match_spot(m.group(1).strip())
        t = _fuzzy_match_spot(m.group(2).strip())
        if f and t:
            return f, t
    m = re.search(
        r"to\s+(.+?)\s+from\s+(.+?)(?:\s+using|\s+with|\s+i have|[,\.\?]|$)",
        text, re.IGNORECASE
    )
    if m:
        t = _fuzzy_match_spot(m.group(1).strip())
        f = _fuzzy_match_spot(m.group(2).strip())
        if f and t:
            return f, t
    spots = _spots_in_order(text)
    if len(spots) >= 2:
        return spots[0], spots[1]
    return None


def _fill_route(
    message: str,
    context: dict,
    from_id: Optional[str],
    to_id: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Fill missing route endpoints from context (one-spot swap or full reuse)."""
    if from_id and to_id:
        return from_id, to_id

    ctx_from, ctx_to = context.get("from_id"), context.get("to_id")
    ctx_spot = context.get("spot_id")

    # Use from_id/to_id from context if available
    if ctx_from and ctx_to:
        msg_spots = _spots_in_order(message)
        text_l = _normalize(message)
        if len(msg_spots) == 1:
            spot = msg_spots[0]
            if re.search(r"\bfrom\b", text_l):
                return (spot, ctx_to) if spot != ctx_to else (spot, ctx_from)
            if re.search(r"\bto\b", text_l):
                return (ctx_from, spot) if spot != ctx_from else (ctx_to, spot)
            return (ctx_from, spot) if spot != ctx_from else (ctx_from, ctx_to)
        return ctx_from, ctx_to

    # If only one location found in message, use spot_id from context as the other
    msg_spots = _spots_in_order(message)
    if len(msg_spots) == 1 and ctx_spot:
        return (ctx_spot, msg_spots[0]) if msg_spots[0] != ctx_spot else (ctx_spot, ctx_spot)

    # If no locations in message but context has a spot, use it as the origin
    if not msg_spots and ctx_spot:
        return (ctx_spot, None)

    return from_id, to_id


# Main engine

class SmartourAI:
    """Rule-based + fuzzy-matching AI engine for Ilocos Sur tourism."""

    def respond(self, message: str, history: List[dict]) -> str:
        """Route a message to a handler: a confident model prediction takes
        priority over the rules; if both are unsure, ask a clarifying question
        (the model's near-miss is used as a hint)."""
        intent = None
        hint = None
        model = get_intent_model()
        if model.is_trained:
            predicted, confidence = model.predict(message)
            if predicted is not None:
                intent = predicted
            else:
                # Not confident enough — remember what it almost said
                top = model.predict_top(message, k=1)
                if top:
                    hint = top[0][0]

        if intent is None:
            intent = _detect_intent(message)

        # Reuse route/car/spot from the last exchange for follow-ups
        context = _extract_context(history)

        # Short follow-ups naming a spot but no intent keywords reuse the last topic
        if intent == "unknown" and (_spots_in_order(message) or _fuzzy_match_spot(message)) \
                and context.get("last_topic") in ("fuel_cost", "distance", "describe"):
            intent = context["last_topic"]

        # A bare car reply ("motorcycle", "and with a van?") completes the
        # pending fuel estimate from memory. "describe" too, since "what about
        # a suv?" hits the "about" keyword first.
        if intent in ("unknown", "recommend", "describe") and _is_bare_car_reply(message) \
                and context.get("last_topic") == "fuel_cost":
            intent = "fuel_cost"

        return self._route(intent, message, hint=hint, context=context)

    def _route(
        self,
        intent: str,
        message: str,
        hint: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> str:
        model = get_intent_model()
        context = context or {}

        # Custom intents trained with a canned response take priority
        if intent in model.custom_responses:
            return model.custom_responses[intent]

        # "unknown" is itself a handler but needs the hint for a targeted prompt
        if intent == "unknown":
            return self._handle_unknown(message, hint=hint)

        handler = getattr(self, f"_handle_{intent}", None)
        if handler is not None:
            return handler(message, context=context)

        # Fallback: try to match a spot name directly (message, then memory)
        spot_id = _fuzzy_match_spot(message) or context.get("spot_id")
        if spot_id:
            return _build_spot_card(spot_id)

        return self._handle_unknown(message, hint=hint)

    # Intent handlers

    def _handle_greeting(self, message: str = "", context: Optional[dict] = None) -> str:
        return (
            "Kumusta! Welcome to Smartour!\n\n"
            "I'm your Ilocos Sur travel assistant. Here's what I can help you with:\n\n"
            "**Find tourist spots** — Ask: _\"What are the best places to visit?\"_\n"
            "**Spot details** — Ask: _\"Tell me about Calle Crisologo\"_\n"
            "**Fuel estimate** — Ask: _\"From Calle Crisologo to UNP, I have a sedan\"_\n"
            "**Distance** — Ask: _\"How far is Bantay Church from Vigan Cathedral?\"_\n\n"
            "What would you like to explore?"
        )

    def _handle_help(self, message: str = "", context: Optional[dict] = None) -> str:
        return (
"**Smartour AI — What I can do:**\n\n"
            "1. **Recommend spots** — \"Suggest heritage spots\" / \"Best places in Ilocos Sur\"\n"
            "2. **Describe a spot** — \"Tell me about Baluarte\" / \"What is Pinsal Falls?\"\n"
            "3. **Fuel cost** — \"From Calle Crisologo to UNP using a SUV\"\n"
            "4. **Distance** — \"How far is Bantay Church from Vigan Cathedral?\"\n\n"
            "**Available car types:** motorcycle, sedan, SUV, van, pickup, multicab\n"
            "**Fuel types:** gasoline (default), diesel, premium\n\n"
            "Try asking anything about Ilocos Sur!"
        )

    def _handle_recommend(self, message: str, context: Optional[dict] = None) -> str:
        category = _detect_category(message)
        ref_muni = _detect_reference_municipality(message, context) or DEFAULT_MUNICIPALITY

        if category and category in CATEGORY_RECOMMENDATIONS:
            spot_ids = CATEGORY_RECOMMENDATIONS[category]
            titles = {
                "heritage": "Top Heritage & Historical Spots",
                "nature": "Nature Spots in Ilocos Sur",
                "beach": "Beach Destinations",
                "food": "Best Food & Cultural Spots",
                "family": "Family-Friendly Spots",
                "adventure": "Adventure Spots",
                "must_see": "Must-See Spots in Ilocos Sur",
            }
            title = titles.get(category, "Recommended Spots")
            return _build_proximity_recommendation(spot_ids, title, ref_muni)

        # Default: show must-see
        return _build_proximity_recommendation(
            CATEGORY_RECOMMENDATIONS["must_see"],
            "Top Must-See Spots in Ilocos Sur",
            ref_muni,
        )

    def _handle_describe(self, message: str, context: Optional[dict] = None) -> str:
        # Conversation memory: "what about its history?" reuses the last spot
        spot_id = _fuzzy_match_spot(message) or (context or {}).get("spot_id")
        if spot_id:
            if _is_directions_query(message):
                return _build_directions(spot_id)
            return _build_spot_card(spot_id)
        return (
            "I couldn't find that specific spot. Try mentioning the spot name more clearly.\n\n"
            "**Known spots:** " + ", ".join(s["name"] for s in TOURIST_SPOTS.values())
        )

    def _handle_fuel_cost(self, message: str, context: Optional[dict] = None) -> str:
        context = context or {}
        car_id = _detect_car_type(message) or context.get("car_id")
        fuel_type = _detect_fuel_type(message)

        from_id, to_id = None, None

        from_match = re.search(
            r"from\s+(.+?)\s+to\s+(.+?)(?:\s+using|\s+with|\s+i have|[,\.\?]|$)",
            message, re.IGNORECASE
        )
        if from_match:
            from_id = _fuzzy_match_spot(from_match.group(1).strip())
            to_id = _fuzzy_match_spot(from_match.group(2).strip())

        # Reversed order: "to X from Y"
        if not from_id or not to_id:
            to_from_match = re.search(
                r"to\s+(.+?)\s+from\s+(.+?)(?:\s+using|\s+with|\s+i have|[,\.\?]|$)",
                message, re.IGNORECASE
            )
            if to_from_match:
                to_id = _fuzzy_match_spot(to_from_match.group(1).strip())
                from_id = _fuzzy_match_spot(to_from_match.group(2).strip())

        if not from_id or not to_id:
            found = _spots_in_order(message)[:2]
            if len(found) == 2:
                from_id, to_id = found[0], found[1]
            elif len(found) == 1:
                ctx_spot = context.get("spot_id")
                if ctx_spot and found[0] != ctx_spot:
                    from_id, to_id = ctx_spot, found[0]

        # Conversation memory: reuse the route from the previous exchange
        if not from_id or not to_id:
            from_id, to_id = _fill_route(message, context, from_id, to_id)

        if not car_id:
            # If the route is known, still give the distance while asking for a car
            if from_id and to_id:
                distance = _estimate_distance(from_id, to_id)
                from_name = TOURIST_SPOTS[from_id]["name"]
                to_name = TOURIST_SPOTS[to_id]["name"]
                return (
                    f"**Distance Estimate**\n"
                    f"From: **{from_name}**\n"
                    f"To: **{to_name}**\n"
                    f"Estimated distance: **{distance} km**\n\n"
                    f"Tell me your **vehicle type** and I'll estimate the fuel cost:\n"
                    f"- Motorcycle, Sedan, SUV, Van, Pickup, Multicab, or Electric\n\n"
                    f"Example: _\"from {from_name} to {to_name} using a sedan\"_"
                )
            return (
                "I need to know your **vehicle type** to estimate fuel cost.\n\n"
                "Please tell me your car type:\n"
                "- Motorcycle, Sedan, SUV, Van, Pickup, Multicab, or Electric\n\n"
                "Example: _\"From Calle Crisologo to Bantay Church using a sedan\"_"
            )

        if not from_id or not to_id:
            # If we have one location from context, ask for the other
            if from_id and not to_id:
                from_name = TOURIST_SPOTS[from_id]["name"]
                return (
                    f"I found your vehicle: **{CAR_TYPES[car_id]['label']}**\n\n"
                    f"From **{from_name}** to where?\n"
                    "Example: _\"From Calle Crisologo to UNP using a sedan\"_"
                )
            return (
                f"I found your vehicle: **{CAR_TYPES[car_id]['label']}**\n\n"
                "But I need **two locations** to estimate fuel cost.\n"
                "Example: _\"From Calle Crisologo to UNP using a sedan\"_\n\n"
                "**Available spots:** " + ", ".join(s["name"] for s in TOURIST_SPOTS.values())
            )

        distance = _estimate_distance(from_id, to_id)
        info = _fuel_cost(distance, car_id, fuel_type)
        from_name = TOURIST_SPOTS[from_id]["name"]
        to_name = TOURIST_SPOTS[to_id]["name"]

        return (
            f"**Trip: {from_name} → {to_name}**\n\n"
            + _build_fuel_response(info)
        )

    def _handle_distance(self, message: str, context: Optional[dict] = None) -> str:
        context = context or {}
        from_id, to_id = None, None

        from_match = re.search(
            r"from\s+(.+?)\s+to\s+(.+?)(?:[,\.\?]|$)",
            message, re.IGNORECASE
        )
        if from_match:
            from_id = _fuzzy_match_spot(from_match.group(1).strip())
            to_id = _fuzzy_match_spot(from_match.group(2).strip())

        # Reversed order: "to X from Y"
        if not from_id or not to_id:
            to_from_match = re.search(
                r"to\s+(.+?)\s+from\s+(.+?)(?:[,\.\?]|$)",
                message, re.IGNORECASE
            )
            if to_from_match:
                to_id = _fuzzy_match_spot(to_from_match.group(1).strip())
                from_id = _fuzzy_match_spot(to_from_match.group(2).strip())

        # Fallback: "how far is X from Y" pattern
        if not from_id or not to_id:
            alt_match = re.search(
                r"(?:how far is|distance.*?)\s+(.+?)\s+(?:from|to)\s+(.+?)(?:[,\.\?]|$)",
                message, re.IGNORECASE
            )
            if alt_match:
                from_id = _fuzzy_match_spot(alt_match.group(1).strip())
                to_id = _fuzzy_match_spot(alt_match.group(2).strip())

        if not from_id or not to_id:
            found = _spots_in_order(message)
            if len(found) >= 2:
                from_id, to_id = found[0], found[1]
            elif len(found) == 1:
                # Only one spot found — use context for the other
                ctx_spot = context.get("spot_id")
                if ctx_spot and found[0] != ctx_spot:
                    from_id, to_id = ctx_spot, found[0]

        # Conversation memory: reuse the route from the previous exchange
        if not from_id or not to_id:
            from_id, to_id = _fill_route(message, context, from_id, to_id)

        if from_id and to_id:
            distance = _estimate_distance(from_id, to_id)
            from_name = TOURIST_SPOTS[from_id]["name"]
            to_name = TOURIST_SPOTS[to_id]["name"]
            reply = (
                f"**Distance Estimate**\n"
                f"From: **{from_name}**\n"
                f"To: **{to_name}**\n"
                f"Estimated distance: **{distance} km** (approximate road distance)"
            )
            # Compound request: also show a fuel estimate when asked
            if _wants_fuel(message):
                reply += "\n\n" + _fuel_estimate_block(from_id, to_id, message, context)
            return reply

        # Only one location from context — ask for the destination
        if from_id and not to_id:
            from_name = TOURIST_SPOTS[from_id]["name"]
            return (
                f"Distance from **{from_name}** to where?\n\n"
                "Example: _\"How far is {from_name} from Bantay Church?\"_"
            )

        return (
            "Please specify two locations.\n"
            "Example: _\"How far is Bantay Church from Calle Crisologo?\"_"
        )

    def _handle_unknown(self, message: str, hint: Optional[str] = None) -> str:
        """Ask a clarifying question instead of a flat "I don't know":
        spot mention → model-hint nudge → general prompt."""
        spot_id = _find_spot_reference(message)
        if spot_id:
            spot = TOURIST_SPOTS[spot_id]
            return (
                f"I think you're asking about **{spot['name']}** — what would you like to know?\n\n"
                f"_\"Tell me about {spot['name']}\"_ for details\n"
                f"_\"Fuel cost from {spot['name']} to [another spot] using a sedan\"_\n"
                f"_\"How far is {spot['name']} from [another spot]?\"_\n"
                f"Or ask me to recommend places around {spot['location']}!"
            )

        if hint and hint in _CLARIFY_HINTS:
            return _CLARIFY_HINTS[hint]

        return (
            "I'm not sure I understood that. I can help you with:\n\n"
            "**Find places** — _\"best places to visit in Vigan\"_\n"
            "**About a spot** — _\"tell me about Calle Crisologo\"_\n"
            "**Fuel cost** — _\"from Calle Crisologo to UNP using a sedan\"_\n"
            "**Distance** — _\"how far is Bantay Church from Plaza Salcedo?\"_\n\n"
            "What would you like to know?"
        )


# Singleton instance
smartour_ai = SmartourAI()
