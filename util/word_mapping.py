'''
Author: Andrew
Date: 2021-12-04 18:19:39
LastEditors: Andrew
LastEditTime: 2022-04-29 17:13:59
FilePath: /COPNERNER/util/word_mapping.py
'''
from collections import OrderedDict

# # Few-NERD
WORD_MAP = OrderedDict()

WORD_MAP['O'] = ['none']
WORD_MAP['E'] = ['entity']

WORD_MAP['location-GPE'] = ['nation']
WORD_MAP['location-bodiesofwater'] = ['water']
WORD_MAP['location-island'] = ['island']
WORD_MAP['location-mountain'] = ['mountain']
WORD_MAP['location-park'] = ['parks']
WORD_MAP['location-road/railway/highway/transit'] = ['road']
WORD_MAP['location-other'] = ['location']

WORD_MAP['person-actor'] = ['actor']
WORD_MAP['person-artist/author'] = ['artist']
WORD_MAP['person-athlete'] = ['athlete']
WORD_MAP['person-director'] = ['director']
WORD_MAP['person-politician'] = ['politician']
WORD_MAP['person-scholar'] = ['scholar']
WORD_MAP['person-soldier'] = ['soldier']
WORD_MAP['person-other'] = ['person']

WORD_MAP['organization-company'] = ['company']
WORD_MAP['organization-education'] = ['education']
WORD_MAP['organization-government/governmentagency'] = ['government']
WORD_MAP['organization-media/newspaper'] = ['media']
WORD_MAP['organization-politicalparty'] = ['parties']
WORD_MAP['organization-religion'] = ['religion']
WORD_MAP['organization-showorganization'] = ['show']
WORD_MAP['organization-sportsleague'] = ['league']
WORD_MAP['organization-sportsteam'] = ['team']
WORD_MAP['organization-other'] = ['organization']

WORD_MAP['building-airport'] = ['airport']
WORD_MAP['building-hospital'] = ['hospital']
WORD_MAP['building-hotel'] = ['hotel']
WORD_MAP['building-library'] = ['library']
WORD_MAP['building-restaurant'] = ['restaurant']
WORD_MAP['building-sportsfacility'] = ['facility']
WORD_MAP['building-theater'] = ['theater']
WORD_MAP['building-other'] = ['building']

WORD_MAP['art-broadcastprogram'] = ['broadcast']
WORD_MAP['art-film'] = ['film']
WORD_MAP['art-music'] = ['music']
WORD_MAP['art-painting'] = ['painting']
WORD_MAP['art-writtenart'] = ['writing']
WORD_MAP['art-other'] = ['art']

WORD_MAP['product-airplane'] = ['airplane']
WORD_MAP['product-car'] = ['car']
WORD_MAP['product-food'] = ['food']
WORD_MAP['product-game'] = ['game']
WORD_MAP['product-ship'] = ['ship']
WORD_MAP['product-software'] = ['software']
WORD_MAP['product-train'] = ['train']
WORD_MAP['product-weapon'] = ['weapon']
WORD_MAP['product-other'] = ['product']

WORD_MAP['event-attack/battle/war/militaryconflict'] = ['war']
WORD_MAP['event-disaster'] = ['disaster']
WORD_MAP['event-election'] = ['election']
WORD_MAP['event-protest'] = ['protest']
WORD_MAP['event-sportsevent'] = ['sport']
WORD_MAP['event-other'] = ['event']

WORD_MAP['other-astronomything'] = ['astronomy']
WORD_MAP['other-award'] = ['award']
WORD_MAP['other-biologything'] = ['biology']
WORD_MAP['other-chemicalthing'] = ['chemistry']
WORD_MAP['other-currency'] = ['currency']
WORD_MAP['other-disease'] = ['disease']
WORD_MAP['other-educationaldegree'] = ['degree']
WORD_MAP['other-god'] = ['god']
WORD_MAP['other-language'] = ['language']
WORD_MAP['other-law'] = ['law']
WORD_MAP['other-livingthing'] = ['organism']
WORD_MAP['other-medical'] = ['medical']


# # OntoNotes
ONTONOTES_WORD_MAP = OrderedDict()

ONTONOTES_WORD_MAP['O'] = ['none']
ONTONOTES_WORD_MAP['E'] = ['entity']

ONTONOTES_WORD_MAP['ORG'] = ['organization']
ONTONOTES_WORD_MAP['NORP'] = ['country']
ONTONOTES_WORD_MAP['ORDINAL'] = ['number']
ONTONOTES_WORD_MAP['WORK_OF_ART'] = ['art']
ONTONOTES_WORD_MAP['QUANTITY'] = ['quantity']
ONTONOTES_WORD_MAP['LAW'] = ['law']
ONTONOTES_WORD_MAP['GPE'] = ['nation']
ONTONOTES_WORD_MAP['CARDINAL'] = ['cardinal']
ONTONOTES_WORD_MAP['PERCENT'] = ['percent']
ONTONOTES_WORD_MAP['TIME'] = ['time']
ONTONOTES_WORD_MAP['EVENT'] = ['event']
ONTONOTES_WORD_MAP['LANGUAGE'] = ['language']
ONTONOTES_WORD_MAP['PERSON'] = ['person']
ONTONOTES_WORD_MAP['DATE'] = ['date']
ONTONOTES_WORD_MAP['MONEY'] = ['money']
ONTONOTES_WORD_MAP['LOC'] = ['location']
ONTONOTES_WORD_MAP['FAC'] = ['facility']
ONTONOTES_WORD_MAP['PRODUCT'] = ['product']


# # OntoNotes
# ONTONOTES_11_WORD_MAP = OrderedDict()

# ONTONOTES_11_WORD_MAP['O'] = ['none']
# ONTONOTES_WORD_MAP['E'] = ['entity']

# ONTONOTES_11_WORD_MAP['WORK_OF_ART'] = ['art']
# ONTONOTES_11_WORD_MAP['ORG'] = ['organization']
# ONTONOTES_11_WORD_MAP['NORP'] = ['country']
# ONTONOTES_11_WORD_MAP['LANGUAGE'] = ['language']
# ONTONOTES_11_WORD_MAP['PERSON'] = ['person']
# ONTONOTES_11_WORD_MAP['LAW'] = ['law']
# ONTONOTES_11_WORD_MAP['EVENT'] = ['event']
# ONTONOTES_11_WORD_MAP['FAC'] = ['facility']
# ONTONOTES_11_WORD_MAP['LOC'] = ['location']
# ONTONOTES_11_WORD_MAP['GPE'] = ['nation']
# ONTONOTES_11_WORD_MAP['PRODUCT'] = ['product']




# # CoNLL 03
CONLL_WORD_MAP = OrderedDict()

CONLL_WORD_MAP['O'] = ['none']
CONLL_WORD_MAP['E'] = ['entity']

CONLL_WORD_MAP['ORG'] = ['organization']
CONLL_WORD_MAP['PER'] = ['person']
CONLL_WORD_MAP['LOC'] = ['location']
CONLL_WORD_MAP['MISC'] = ['other']






# # WNUT
WNUT_WORD_MAP = OrderedDict()

WNUT_WORD_MAP['O'] = ['none']
WNUT_WORD_MAP['E'] = ['entity']

WNUT_WORD_MAP['location'] = ['location']
WNUT_WORD_MAP['group'] = ['group']
WNUT_WORD_MAP['corporation'] = ['company']
WNUT_WORD_MAP['person'] = ['person']
WNUT_WORD_MAP['creative-work'] = ['creativity']
WNUT_WORD_MAP['product'] = ['product']






# # I2B2
I2B2_WORD_MAP = OrderedDict()

I2B2_WORD_MAP['O'] = ['none']
I2B2_WORD_MAP['E'] = ['entity']

I2B2_WORD_MAP['DATE'] = ['date'] 
I2B2_WORD_MAP['AGE'] = ['age'] 
I2B2_WORD_MAP['STATE'] = ['state'] 
I2B2_WORD_MAP['PATIENT'] = ['people']
I2B2_WORD_MAP['DOCTOR'] = ['doctor']
I2B2_WORD_MAP['MEDICALRECORD'] = ['number']
I2B2_WORD_MAP['HOSPITAL'] = ['hospital']
I2B2_WORD_MAP['PHONE'] = ['phone']
I2B2_WORD_MAP['IDNUM'] = ['id']
I2B2_WORD_MAP['USERNAME'] = ['name']
I2B2_WORD_MAP['STREET'] = ['street']
I2B2_WORD_MAP['CITY'] = ['city']
I2B2_WORD_MAP['ZIP'] = ['zip']
I2B2_WORD_MAP['EMAIL'] = ['email']
I2B2_WORD_MAP['PROFESSION'] = ['profession']
I2B2_WORD_MAP['COUNTRY'] = ['country']
I2B2_WORD_MAP['ORGANIZATION'] = ['organization']
I2B2_WORD_MAP['FAX'] = ['ip']

# I2B2_WORD_MAP['LOCATION-OTHER'] = ['location'] 
# I2B2_WORD_MAP['DEVICE'] = ['device'] 
# I2B2_WORD_MAP['BIOID'] = ['bio']
# I2B2_WORD_MAP['HEALTHPLAN'] = ['plan']
# I2B2_WORD_MAP['URL'] = ['link']





# # MIT-Movies
MOVIES_WORD_MAP = OrderedDict()

MOVIES_WORD_MAP['O'] = ['none']
MOVIES_WORD_MAP['E'] = ['entity']

MOVIES_WORD_MAP['CHARACTER'] = ['character']
MOVIES_WORD_MAP['GENRE'] = ['genre']
MOVIES_WORD_MAP['TITLE'] = ['title']
MOVIES_WORD_MAP['PLOT'] = ['plot']
MOVIES_WORD_MAP['RATING'] = ['rating']
MOVIES_WORD_MAP['YEAR'] = ['year']

MOVIES_WORD_MAP['REVIEW'] = ['review']
MOVIES_WORD_MAP['ACTOR'] = ['actor']
MOVIES_WORD_MAP['DIRECTOR'] = ['director']
MOVIES_WORD_MAP['SONG'] = ['song']
MOVIES_WORD_MAP['RATINGS_AVERAGE'] = ['average']
MOVIES_WORD_MAP['TRAILER'] = ['trailer']
