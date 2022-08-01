'''
Author: Andrew
Date: 2021-12-04 18:19:39
LastEditors: Andrew
LastEditTime: 2022-04-29 17:13:59
FilePath: /COPNERNER/util/word_mapping.py
'''
from collections import OrderedDict

# # Few-NERD
WORD_MAP_SYN = OrderedDict()

WORD_MAP_SYN['O'] = ['none', 'other', 'others']
WORD_MAP_SYN['E'] = ['entity', 'entity', 'entity']

WORD_MAP_SYN['location-GPE'] = ['nation', 'organizations', 'administration']
WORD_MAP_SYN['location-bodiesofwater'] = ['water', 'river', 'lake']
WORD_MAP_SYN['location-island'] = ['island', 'isle', 'island']
WORD_MAP_SYN['location-mountain'] = ['mountain', 'hill', 'peak']
WORD_MAP_SYN['location-park'] = ['parks', 'yard', 'field']
WORD_MAP_SYN['location-road/railway/highway/transit'] = ['road', 'highway', 'transit']
WORD_MAP_SYN['location-other'] = ['location', 'location', 'location']

WORD_MAP_SYN['person-actor'] = ['actor','performer','player']
WORD_MAP_SYN['person-artist/author'] = ['artist', 'master', 'creator']
WORD_MAP_SYN['person-athlete'] = ['athlete', 'contestant', 'sportsman']
WORD_MAP_SYN['person-director'] = ['director', 'supervisor', 'organizer']
WORD_MAP_SYN['person-politician'] = ['politician', 'legislator', 'senator']
WORD_MAP_SYN['person-scholar'] = ['scholar', 'academic', 'scholastic']
WORD_MAP_SYN['person-soldier'] = ['soldier', 'fighter', 'warrior']
WORD_MAP_SYN['person-other'] = ['person', 'person', 'person']

WORD_MAP_SYN['organization-company'] = ['company', 'enterprise', 'corporation']
WORD_MAP_SYN['organization-education'] = ['education', 'school', 'institute']
WORD_MAP_SYN['organization-government/governmentagency'] = ['government', 'regime', 'administration']
WORD_MAP_SYN['organization-media/newspaper'] = ['media', 'newspaper', 'podcast']
WORD_MAP_SYN['organization-politicalparty'] = ['party', 'organization', 'confederacy']
WORD_MAP_SYN['organization-religion'] = ['religion', 'creed', 'cult']
WORD_MAP_SYN['organization-showorganization'] = ['show', 'demonstration', 'illustration']
WORD_MAP_SYN['organization-sportsleague'] = ['league', 'group', 'association']
WORD_MAP_SYN['organization-sportsteam'] = ['team', 'crew', 'gang']
WORD_MAP_SYN['organization-other'] = ['organization', 'organization', 'organization']

WORD_MAP_SYN['building-airport'] = ['airport', 'airport', 'airfield']
WORD_MAP_SYN['building-hospital'] = ['hospital', 'clinic', 'infirmary']
WORD_MAP_SYN['building-hotel'] = ['hotel', 'hostel', 'inn']
WORD_MAP_SYN['building-library'] = ['library','libraries', 'library']
WORD_MAP_SYN['building-restaurant'] = ['restaurant', 'cafe', 'restaurant']
WORD_MAP_SYN['building-sportsfacility'] = ['facility', 'sports', 'equipment']
WORD_MAP_SYN['building-theater'] = ['theater', 'cinema', 'playhouse']
WORD_MAP_SYN['building-other'] = ['building', 'building', 'building']

WORD_MAP_SYN['art-broadcastprogram'] = ['broadcast', 'programme', 'show']
WORD_MAP_SYN['art-film'] = ['film', 'movie', 'cinema']
WORD_MAP_SYN['art-music'] = ['music','song', 'composition']
WORD_MAP_SYN['art-painting'] = ['painting', 'canvas', 'picture']
WORD_MAP_SYN['art-writtenart'] = ['writing', 'document', 'book']
WORD_MAP_SYN['art-other'] = ['art', 'art', 'art']

WORD_MAP_SYN['product-airplane'] = ['airplane', 'plane', 'aircraft']
WORD_MAP_SYN['product-car'] = ['car', 'automobile', 'motor']
WORD_MAP_SYN['product-food'] = ['food', 'edible', 'eating']
WORD_MAP_SYN['product-game'] = ['game', 'amusement', 'entertainment']
WORD_MAP_SYN['product-ship'] = ['ship', 'vessel', 'boat']
WORD_MAP_SYN['product-software'] = ['software', 'application', 'program']
WORD_MAP_SYN['product-train'] = ['train', 'locomotive', 'railway']
WORD_MAP_SYN['product-weapon'] = ['weapon', 'arms', 'artillery']
WORD_MAP_SYN['product-other'] = ['product','product','product']

WORD_MAP_SYN['event-attack/battle/war/militaryconflict'] = ['war', 'battle', 'attack']
WORD_MAP_SYN['event-disaster'] = ['disaster', 'disasters', 'disaster']
WORD_MAP_SYN['event-election'] = ['election', 'selection', 'choice']
WORD_MAP_SYN['event-protest'] = ['protest', 'demur', 'objection']
WORD_MAP_SYN['event-sportsevent'] = ['sport', 'game', 'competition']
WORD_MAP_SYN['event-other'] = ['event','event','event']

WORD_MAP_SYN['other-astronomything'] = ['astronomy', 'astronomy', 'astronomy']
WORD_MAP_SYN['other-award'] = ['award', 'prize', 'grant']
WORD_MAP_SYN['other-biologything'] = ['biology', 'biology', 'biology']
WORD_MAP_SYN['other-chemicalthing'] = ['chemistry', 'chemical', 'substance']
WORD_MAP_SYN['other-currency'] = ['currency', 'money', 'cash']
WORD_MAP_SYN['other-disease'] = ['disease', 'illness', 'diseases']
WORD_MAP_SYN['other-educationaldegree'] = ['degree', 'diploma', 'title']
WORD_MAP_SYN['other-god'] = ['god', 'deity', 'divinity']
WORD_MAP_SYN['other-language'] = ['language', 'tongue', 'dialect']
WORD_MAP_SYN['other-law'] = ['law', 'statute', 'ordinance']
WORD_MAP_SYN['other-livingthing'] = ['organism', 'creature', 'being']
WORD_MAP_SYN['other-medical'] = ['medical', 'pharmaceutical', 'medication']


# # OntoNotes
ONTONOTES_WORD_MAP_SYN = OrderedDict()

ONTONOTES_WORD_MAP_SYN['O'] = ['none', 'other', 'others']
ONTONOTES_WORD_MAP_SYN['E'] = ['entity', 'entity', 'entity']

ONTONOTES_WORD_MAP_SYN['ORG'] = ['organization', 'association', 'federation']
ONTONOTES_WORD_MAP_SYN['NORP'] = ['nationality', 'religion', 'politics']
ONTONOTES_WORD_MAP_SYN['ORDINAL'] = ['number', 'ordinal', 'numeral']
ONTONOTES_WORD_MAP_SYN['WORK_OF_ART'] = ['art', 'artwork', 'creativity']
ONTONOTES_WORD_MAP_SYN['QUANTITY'] = ['quantity', 'amount', 'volume']
ONTONOTES_WORD_MAP_SYN['LAW'] = ['law', 'statute', 'ordinance']
ONTONOTES_WORD_MAP_SYN['GPE'] = ['country', 'nation', 'state']
ONTONOTES_WORD_MAP_SYN['CARDINAL'] = ['cardinal', 'number', 'numeral']
ONTONOTES_WORD_MAP_SYN['PERCENT'] = ['percent', 'percentage', 'proportion']
ONTONOTES_WORD_MAP_SYN['TIME'] = ['time', 'when', 'period']
ONTONOTES_WORD_MAP_SYN['EVENT'] = ['event', 'incident', 'occurance']
ONTONOTES_WORD_MAP_SYN['LANGUAGE'] = ['language', 'tongue', 'dialect']
ONTONOTES_WORD_MAP_SYN['PERSON'] = ['person', 'man', 'individual']
ONTONOTES_WORD_MAP_SYN['DATE'] = ['date', 'day', 'time']
ONTONOTES_WORD_MAP_SYN['MONEY'] = ['money', 'currency', 'cash']
ONTONOTES_WORD_MAP_SYN['LOC'] = ['location', 'place', 'position']
ONTONOTES_WORD_MAP_SYN['FAC'] = ['facility', 'amenity', 'equipment']
ONTONOTES_WORD_MAP_SYN['PRODUCT'] = ['product']



# ONTONOTES_11_WORD_MAP_SYN = OrderedDict()

# ONTONOTES_11_WORD_MAP_SYN['O'] = ['none', 'other', 'others']
# # ONTONOTES_11_WORD_MAP_SYN['E'] = ['entity', 'entity', 'entity']

# ONTONOTES_11_WORD_MAP_SYN['WORK_OF_ART'] = ['art', 'artwork', 'creativity']
# ONTONOTES_11_WORD_MAP_SYN['ORG'] = ['organization', 'association', 'federation']
# ONTONOTES_11_WORD_MAP_SYN['NORP'] = ['nationality', 'religion', 'politics']
# ONTONOTES_11_WORD_MAP_SYN['LANGUAGE'] = ['language', 'tongue', 'dialect']
# ONTONOTES_11_WORD_MAP_SYN['PERSON'] = ['person', 'man', 'individual']
# ONTONOTES_11_WORD_MAP_SYN['LAW'] = ['law', 'statute', 'ordinance']
# ONTONOTES_11_WORD_MAP_SYN['EVENT'] = ['event', 'incident', 'occurance']
# ONTONOTES_11_WORD_MAP_SYN['FAC'] = ['facility', 'amenity', 'equipment']
# ONTONOTES_11_WORD_MAP_SYN['LOC'] = ['location', 'place', 'position']
# ONTONOTES_11_WORD_MAP_SYN['GPE'] = ['country', 'nation', 'state']
# ONTONOTES_11_WORD_MAP_SYN['PRODUCT'] = ['product']




# # CoNLL 03
CONLL_WORD_MAP_SYN = OrderedDict()

CONLL_WORD_MAP_SYN['O'] = ['none', 'other', 'others']
CONLL_WORD_MAP_SYN['E'] = ['entity', 'entity', 'entity']

CONLL_WORD_MAP_SYN['ORG'] = ['organization', 'association', 'federation']
CONLL_WORD_MAP_SYN['PER'] = ['person', 'man', 'individual']
CONLL_WORD_MAP_SYN['LOC'] = ['location', 'place', 'position']
CONLL_WORD_MAP_SYN['MISC'] = ['miscellaneous', 'miscellaneous', 'miscellaneous']


# # WNUT
WNUT_WORD_MAP_SYN = OrderedDict()

WNUT_WORD_MAP_SYN['O'] = ['none', 'other', 'others']
WNUT_WORD_MAP_SYN['E'] = ['entity', 'entity', 'entity']

WNUT_WORD_MAP_SYN['location'] = ['location', 'place', 'position']
WNUT_WORD_MAP_SYN['group'] = ['group', 'team', 'clique']
WNUT_WORD_MAP_SYN['corporation'] = ['company', 'enterprise', 'firm']
WNUT_WORD_MAP_SYN['person'] = ['person', 'man', 'individual']
WNUT_WORD_MAP_SYN['creative-work'] = ['creativity', 'innovation', 'inspiration']
WNUT_WORD_MAP_SYN['product'] = ['product', 'goods', 'commodity']


# # I2B2
I2B2_WORD_MAP_SYN = OrderedDict()

I2B2_WORD_MAP_SYN['O'] = ['none', 'other', 'others']
I2B2_WORD_MAP_SYN['E'] = ['entity', 'entity', 'entity']

I2B2_WORD_MAP_SYN['DATE'] = ['date', 'day', 'time'] #
I2B2_WORD_MAP_SYN['AGE'] = ['age', 'years', 'old'] #
I2B2_WORD_MAP_SYN['STATE'] = ['state', 'province', 'region'] #
I2B2_WORD_MAP_SYN['PATIENT'] = ['people', 'patient', 'sufferer']
I2B2_WORD_MAP_SYN['DOCTOR'] = ['doctor', 'clinician', 'physician']
I2B2_WORD_MAP_SYN['MEDICALRECORD'] = ['number', 'record', 'log']
I2B2_WORD_MAP_SYN['HOSPITAL'] = ['hospital', 'clinic', 'infirmary']
I2B2_WORD_MAP_SYN['PHONE'] = ['phone', 'telephone', 'number']
I2B2_WORD_MAP_SYN['IDNUM'] = ['id', 'number', 'identification']
I2B2_WORD_MAP_SYN['USERNAME'] = ['name', 'user', 'username']
I2B2_WORD_MAP_SYN['STREET'] = ['street','road', 'lane']
I2B2_WORD_MAP_SYN['CITY'] = ['city', 'town', 'metropolitan']
I2B2_WORD_MAP_SYN['ZIP'] = ['zip', 'zipcode', 'postcode']
I2B2_WORD_MAP_SYN['EMAIL'] = ['email', 'netmail', 'e-mail']
I2B2_WORD_MAP_SYN['PROFESSION'] = ['profession', 'occupation', 'position']
I2B2_WORD_MAP_SYN['COUNTRY'] = ['country', 'nation', 'kingdom']
I2B2_WORD_MAP_SYN['ORGANIZATION'] = ['organization', 'association', 'federation']
I2B2_WORD_MAP_SYN['FAX'] = ['ip', 'fax', 'number']

# I2B2_WORD_MAP_SYN['LOCATION-OTHER'] = ['location'] 
# I2B2_WORD_MAP_SYN['DEVICE'] = ['device', 'instrument', 'machine'] 
# I2B2_WORD_MAP_SYN['BIOID'] = ['bio', 'number','id']
# I2B2_WORD_MAP_SYN['HEALTHPLAN'] = ['plan', 'intend', 'proposal']
# I2B2_WORD_MAP_SYN['URL'] = ['url', 'link', 'address']

# # MIT-Movies
MOVIES_WORD_MAP_SYN = OrderedDict()

MOVIES_WORD_MAP_SYN['O'] = ['none', 'other', 'others']
MOVIES_WORD_MAP_SYN['E'] = ['entity', 'entity', 'entity']

MOVIES_WORD_MAP_SYN['CHARACTER'] = ['character', 'casting', 'role']
MOVIES_WORD_MAP_SYN['GENRE'] = ['genre', 'variety', 'kind']
MOVIES_WORD_MAP_SYN['TITLE'] = ['title', 'heading', 'name']
MOVIES_WORD_MAP_SYN['PLOT'] = ['plot', 'scenario', 'narrative']
MOVIES_WORD_MAP_SYN['RATING'] = ['rating', 'assessment', 'rank']
MOVIES_WORD_MAP_SYN['YEAR'] = ['year', 'time', 'period']

MOVIES_WORD_MAP_SYN['REVIEW'] = ['review', 'comment', 'remark']
MOVIES_WORD_MAP_SYN['ACTOR'] = ['actor', 'player', 'actress']
MOVIES_WORD_MAP_SYN['DIRECTOR'] = ['director', 'supervisor', 'organizer']
MOVIES_WORD_MAP_SYN['SONG'] = ['song', 'vocal', 'music']
MOVIES_WORD_MAP_SYN['RATINGS_AVERAGE'] = ['average', 'mean', 'middle']
MOVIES_WORD_MAP_SYN['TRAILER'] = ['trailer', 'preview', 'advertisement']


