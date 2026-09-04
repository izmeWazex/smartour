CATEGORY_KEYWORDS = {
    "historical": ["historical", "heritage", "ancestral", "old", "church", "museum"],
    "beach": ["beach", "swim", "shore", "sand"],
    "food": ["food", "eat", "restaurant", "bagnet", "empanada", "hungry" , "craving"],
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
    # print(extract_category("I want a peaceful historical place"))
    # print(extract_category("gusto ko kumain ng bagnet"))
    # print(extract_category("suggest me a good beach"))
    # print(extract_category("asdkjaskjd random text"))

    # trickier / ambiguous cases
    print(extract_category("Where can I dine near the ancestral houses?"))   # food + historical both present
    print(extract_category("I'm craving something near the shore"))          # food + beach both present
    print(extract_category("Is there a restaurant inside an old museum?"))   # food + historical both present
    print(extract_category("Pwede ba mag hike papunta sa waterfalls?"))      # nature, mixed English/Tagalog
    print(extract_category("HISTORICAL PLACES i can eat PLEASE"))                     # all caps — tests .lower()
    print(extract_category(""))                                             # empty string
    print(extract_category("beachhh vibes only"))                          # "beach" is a substring, should still match
    print(extract_category("I heard bagnet is famous near the beach"))     # food + beach both present
    print(extract_category("is there a waterfall in vigan city")) 