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

WORD_MAP['O']= ['none']
WORD_MAP['E']= ['disease']

WORD_MAP['location-GPE']=['organism']
WORD_MAP['location-bodiesofwater']=['biology']
WORD_MAP['location-island']=['food']
WORD_MAP['location-mountain']=['scholar']
WORD_MAP['location-park']=['product']
WORD_MAP['location-road/railway/highway/transit']=['airport']
WORD_MAP['location-other']=['government']
WORD_MAP['person-actor']=['art']
WORD_MAP['person-artist/author']=['law']
WORD_MAP['person-athlete']=['event']
WORD_MAP['person-director']=['hospital']
WORD_MAP['person-politician']=['media']
WORD_MAP['person-scholar']=['disaster']
WORD_MAP['person-soldier']=['hotel']
WORD_MAP['person-other']=['music']
WORD_MAP['organization-company']=['broadcast']
WORD_MAP['organization-education']=['parks']
WORD_MAP['organization-government/governmentagency']=['restaurant']
WORD_MAP['organization-media/newspaper']=['medical']
WORD_MAP['organization-politicalparty']=['artist']
WORD_MAP['organization-religion']=['building']
WORD_MAP['organization-showorganization']=['god']
WORD_MAP['organization-sportsleague']=['car']
WORD_MAP['organization-sportsteam']=['nation']
WORD_MAP['organization-other']=['mountain']
WORD_MAP['building-airport']=['war']
WORD_MAP['building-hospital']=['water']
WORD_MAP['building-hotel']=['film']
WORD_MAP['building-library']=['chemistry']
WORD_MAP['building-restaurant']=['currency']
WORD_MAP['building-sportsfacility']=['religion']
WORD_MAP['building-theater']=['ship']
WORD_MAP['building-other']=['company']
WORD_MAP['art-broadcastprogram']=['degree']
WORD_MAP['art-film']=['road']
WORD_MAP['art-music']=['game']
WORD_MAP['art-painting']=['location']
WORD_MAP['art-writtenart']=['software']
WORD_MAP['art-other']=['theater']
WORD_MAP['product-airplane']=['election']
WORD_MAP['product-car']=['astronomy']
WORD_MAP['product-food']=['organization']
WORD_MAP['product-game']=['soldier']
WORD_MAP['product-ship']=['facility']
WORD_MAP['product-software']=['language']
WORD_MAP['product-train']=['show']
WORD_MAP['product-weapon']=['disease']
WORD_MAP['product-other']=['education']
WORD_MAP['event-attack/battle/war/militaryconflict']=['award']
WORD_MAP['event-disaster']=['train']
WORD_MAP['event-election']=['weapon']
WORD_MAP['event-protest']=['writing']
WORD_MAP['event-sportsevent']=['actor']
WORD_MAP['event-other']=['airplane']
WORD_MAP['other-astronomything']=['league']
WORD_MAP['other-award']=['parties']
WORD_MAP['other-biologything']=['athlete']
WORD_MAP['other-chemicalthing']=['director']
WORD_MAP['other-currency']=['island']
WORD_MAP['other-disease']=['painting']
WORD_MAP['other-educationaldegree']=['politician']
WORD_MAP['other-god']=['person']
WORD_MAP['other-language']=['library']
WORD_MAP['other-law']=['sport']
WORD_MAP['other-livingthing']=['team']
WORD_MAP['other-medical']=['protest']



# # OntoNotes
ONTONOTES_WORD_MAP = OrderedDict()

ONTONOTES_WORD_MAP['O'] = ['none']
ONTONOTES_WORD_MAP['E'] = ['entity']

ONTONOTES_WORD_MAP['ORG'] = ['product']
ONTONOTES_WORD_MAP['NORP'] = ['organization']
ONTONOTES_WORD_MAP['ORDINAL'] = ['country']
ONTONOTES_WORD_MAP['WORK_OF_ART'] = ['number']
ONTONOTES_WORD_MAP['QUANTITY'] = ['art']
ONTONOTES_WORD_MAP['LAW'] = ['quantity']
ONTONOTES_WORD_MAP['GPE'] = ['law']
ONTONOTES_WORD_MAP['CARDINAL'] = ['nation']
ONTONOTES_WORD_MAP['PERCENT'] = ['cardinal']
ONTONOTES_WORD_MAP['TIME'] = ['percent']
ONTONOTES_WORD_MAP['EVENT'] = ['time']
ONTONOTES_WORD_MAP['LANGUAGE'] = ['event']
ONTONOTES_WORD_MAP['PERSON'] = ['language']
ONTONOTES_WORD_MAP['DATE'] = ['person']
ONTONOTES_WORD_MAP['MONEY'] = ['date']
ONTONOTES_WORD_MAP['LOC'] = ['money']
ONTONOTES_WORD_MAP['FAC'] = ['location']
ONTONOTES_WORD_MAP['PRODUCT'] = ['facility']






# # CoNLL 03
CONLL_WORD_MAP = OrderedDict()

CONLL_WORD_MAP['O'] = ['none']
CONLL_WORD_MAP['E'] = ['entity']

CONLL_WORD_MAP['ORG'] = ['other']
CONLL_WORD_MAP['PER'] = ['organization']
CONLL_WORD_MAP['LOC'] = ['person']
CONLL_WORD_MAP['MISC'] = ['location']





# # WNUT
WNUT_WORD_MAP = OrderedDict()

WNUT_WORD_MAP['O'] = ['none']
WNUT_WORD_MAP['E'] = ['entity']

WNUT_WORD_MAP['location'] = ['product']
WNUT_WORD_MAP['group'] = ['location']
WNUT_WORD_MAP['corporation'] = ['group']
WNUT_WORD_MAP['person'] = ['company']
WNUT_WORD_MAP['creative-work'] = ['person']
WNUT_WORD_MAP['product'] = ['creativity']






# # I2B2
I2B2_WORD_MAP = OrderedDict()

I2B2_WORD_MAP['O'] = ['none']
I2B2_WORD_MAP['E'] = ['entity']

I2B2_WORD_MAP['DATE'] = ['ip']
I2B2_WORD_MAP['AGE'] =  ['date'] 
I2B2_WORD_MAP['STATE'] = ['age'] 
I2B2_WORD_MAP['PATIENT'] = ['state'] 
I2B2_WORD_MAP['DOCTOR'] = ['people']
I2B2_WORD_MAP['MEDICALRECORD'] = ['doctor']
I2B2_WORD_MAP['HOSPITAL'] = ['number']
I2B2_WORD_MAP['PHONE'] = ['hospital']
I2B2_WORD_MAP['IDNUM'] = ['phone']
I2B2_WORD_MAP['USERNAME'] = ['id']
I2B2_WORD_MAP['STREET'] = ['name']
I2B2_WORD_MAP['CITY'] = ['street']
I2B2_WORD_MAP['ZIP'] = ['city']
I2B2_WORD_MAP['EMAIL'] = ['zip']
I2B2_WORD_MAP['PROFESSION'] = ['email']
I2B2_WORD_MAP['COUNTRY'] = ['profession']
I2B2_WORD_MAP['ORGANIZATION'] = ['country']
I2B2_WORD_MAP['FAX'] = ['organization']





# # MIT-Movies
MOVIES_WORD_MAP = OrderedDict()

MOVIES_WORD_MAP['O'] = ['none']
MOVIES_WORD_MAP['E'] = ['entity']

MOVIES_WORD_MAP['CHARACTER'] = ['trailer']
MOVIES_WORD_MAP['GENRE'] = ['character']
MOVIES_WORD_MAP['TITLE'] = ['genre']
MOVIES_WORD_MAP['PLOT'] = ['title']
MOVIES_WORD_MAP['RATING'] = ['plot']
MOVIES_WORD_MAP['YEAR'] = ['rating']

MOVIES_WORD_MAP['REVIEW'] = ['year']
MOVIES_WORD_MAP['ACTOR'] = ['review']
MOVIES_WORD_MAP['DIRECTOR'] = ['actor']
MOVIES_WORD_MAP['SONG'] = ['director']
MOVIES_WORD_MAP['RATINGS_AVERAGE'] = ['song']
MOVIES_WORD_MAP['TRAILER'] = ['average']

