"""Constants and encodings used in regulation scraping and parsing."""

REGULATION_CODES = {
    "type": {
        "UU": "01",
        "PP": "02",
        "PERMENKOMINFO": "03"
    },
    "section": {
        "document": "1",
        "consideration": "2",
        "observation": "3",
        "definition": "4",
        "article": "5"
    }
}

OL_TYPES = {
    "a": "lower-alpha", 
    "lower-alpha": "lower-alpha",
    "decimal": "decimal"
}

WORD_TO_NUMBER = {
    "kesatu": 1,
    "kedua": 2,
    "ketiga": 3,
    "keempat": 4,
    "kelima": 5,
    "keenam": 6,
    "ketujuh": 7,
    "kedelapan": 8,
    "kesembilan": 9,
    "kesepuluh": 10,
    "kesebelas": 11,
    "kedua belas": 12,
    "ketiga belas": 13,
    "keempat belas": 14,
    "kelima belas": 15,
    "keenam belas": 16,
    "ketujuh belas": 17,
    "kedelapan belas": 18,
    "kesembilan belas": 19,
    "kedua puluh": 20,
    "kedua puluh satu": 21,
    "kedua puluh dua": 22,
    "kedua puluh tiga": 23,
    "kedua puluh empat": 24,
    "kedua puluh lima": 25,
    "kedua puluh enam": 26,
    "kedua puluh tujuh": 27,
    "kedua puluh delapan": 28,
    "kedua puluh sembilan": 29,
    "ketiga puluh": 30,
    "ketiga puluh satu": 31,
    "ketiga puluh dua": 32,
    "ketiga puluh tiga": 33,
    "ketiga puluh empat": 34,
    "ketiga puluh lima": 35,
    "ketiga puluh enam": 36,
    "ketiga puluh tujuh": 37,
    "ketiga puluh delapan": 38,
    "ketiga puluh sembilan": 39,
    "keempat puluh": 40,
    "keempat puluh satu": 41,
    "keempat puluh dua": 42,
    "keempat puluh tiga": 43,
    "keempat puluh empat": 44,
    "keempat puluh lima": 45,
    "keempat puluh enam": 46,
    "keempat puluh tujuh": 47,
    "keempat puluh delapan": 48,
    "keempat puluh sembilan": 49,
    "kelima puluh": 50
}
