CATEGORY_KEYWORDS = {
    "historical": ["historical", "heritage", "ancestral", "old", "church", "museum"],
    "beach": ["beach", "swim", "shore", "sand"],
    "food": ["food", "eat", "dine", "restaurant", "bagnet", "empanada", "hungry", "craving"],
    "nature": ["waterfall", "falls", "nature", "hike", "mountain"]
}

def extract_category(user_message: str ) -> list[str] | None:
    message = user_message.lower()
    match =[]

    for category , keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in message for keyword in keywords):
            match.append(category)

    return match if match else None


if __name__ == "__main__":
    print(extract_category("Where can I dine near the ancestral houses?"))
    print(extract_category("I'm craving something near the shore"))
    print(extract_category("Is there a restaurant inside an old museum?"))
    print(extract_category("Pwede ba mag hike papunta sa waterfalls?"))
    print(extract_category("HISTORICAL PLACES i can eat PLEASE"))
    print(extract_category(""))
    print(extract_category("beachhh vibes only"))
    print(extract_category("I heard bagnet is famous near the beach"))
    print(extract_category("is there a waterfall in vigan city"))