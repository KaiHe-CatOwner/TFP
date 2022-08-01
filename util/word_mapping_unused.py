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
WORD_MAP['E']= ['[unused1]']

WORD_MAP['location-GPE']=['[unused2]']
WORD_MAP['location-bodiesofwater']=['[unused3]']
WORD_MAP['location-island']=['[unused4]']
WORD_MAP['location-mountain']=['[unused5]']
WORD_MAP['location-park']=['[unused6]']
WORD_MAP['location-road/railway/highway/transit']=['[unused7]']
WORD_MAP['location-other']=['[unused8]']
WORD_MAP['person-actor']=['[unused9]']
WORD_MAP['person-artist/author']=['[unused10]']
WORD_MAP['person-athlete']=['[unused11]']
WORD_MAP['person-director']=['[unused12]']
WORD_MAP['person-politician']=['[unused13]']
WORD_MAP['person-scholar']=['[unused14]']
WORD_MAP['person-soldier']=['[unused15]']
WORD_MAP['person-other']=['[unused16]']
WORD_MAP['organization-company']=['[unused17]']
WORD_MAP['organization-education']=['[unused18]']
WORD_MAP['organization-government/governmentagency']=['[unused19]']
WORD_MAP['organization-media/newspaper']=['[unused20]']
WORD_MAP['organization-politicalparty']=['[unused21]']
WORD_MAP['organization-religion']=['[unused22]']
WORD_MAP['organization-showorganization']=['[unused23]']
WORD_MAP['organization-sportsleague']=['[unused24]']
WORD_MAP['organization-sportsteam']=['[unused25]']
WORD_MAP['organization-other']=['[unused26]']
WORD_MAP['building-airport']=['[unused27]']
WORD_MAP['building-hospital']=['[unused28]']
WORD_MAP['building-hotel']=['[unused29]']
WORD_MAP['building-library']=['[unused30]']
WORD_MAP['building-restaurant']=['[unused31]']
WORD_MAP['building-sportsfacility']=['[unused32]']
WORD_MAP['building-theater']=['[unused33]']
WORD_MAP['building-other']=['[unused34]']
WORD_MAP['art-broadcastprogram']=['[unused35]']
WORD_MAP['art-film']=['[unused36]']
WORD_MAP['art-music']=['[unused37]']
WORD_MAP['art-painting']=['[unused38]']
WORD_MAP['art-writtenart']=['[unused39]']
WORD_MAP['art-other']=['[unused40]']
WORD_MAP['product-airplane']=['[unused41]']
WORD_MAP['product-car']=['[unused42]']
WORD_MAP['product-food']=['[unused43]']
WORD_MAP['product-game']=['[unused44]']
WORD_MAP['product-ship']=['[unused45]']
WORD_MAP['product-software']=['[unused46]']
WORD_MAP['product-train']=['[unused47]']
WORD_MAP['product-weapon']=['[unused48]']
WORD_MAP['product-other']=['[unused49]']
WORD_MAP['event-attack/battle/war/militaryconflict']=['[unused50]']
WORD_MAP['event-disaster']=['[unused51]']
WORD_MAP['event-election']=['[unused52]']
WORD_MAP['event-protest']=['[unused53]']
WORD_MAP['event-sportsevent']=['[unused54]']
WORD_MAP['event-other']=['[unused55]']
WORD_MAP['other-astronomything']=['[unused56]']
WORD_MAP['other-award']=['[unused57]']
WORD_MAP['other-biologything']=['[unused58]']
WORD_MAP['other-chemicalthing']=['[unused59]']
WORD_MAP['other-currency']=['[unused60]']
WORD_MAP['other-disease']=['[unused61]']
WORD_MAP['other-educationaldegree']=['[unused62]']
WORD_MAP['other-god']=['[unused63]']
WORD_MAP['other-language']=['[unused64]']
WORD_MAP['other-law']=['[unused65]']
WORD_MAP['other-livingthing']=['[unused66]']
WORD_MAP['other-medical']=['[unused67]']



# # OntoNotes
ONTONOTES_WORD_MAP = OrderedDict()

ONTONOTES_WORD_MAP['O'] = ['none']
ONTONOTES_WORD_MAP['E'] = ['[unused1]']

ONTONOTES_WORD_MAP['ORG'] = ['[unused2]']
ONTONOTES_WORD_MAP['NORP'] =['[unused3]']
ONTONOTES_WORD_MAP['ORDINAL'] =  ['[unused4]']
ONTONOTES_WORD_MAP['WORK_OF_ART'] = ['[unused5]']
ONTONOTES_WORD_MAP['QUANTITY'] =['[unused6]']
ONTONOTES_WORD_MAP['LAW'] =  ['[unused7]'] 
ONTONOTES_WORD_MAP['GPE'] = ['[unused8]'] 
ONTONOTES_WORD_MAP['CARDINAL'] = ['[unused9]'] 
ONTONOTES_WORD_MAP['PERCENT'] = ['[unused10]'] 
ONTONOTES_WORD_MAP['TIME'] = ['[unused11]']
ONTONOTES_WORD_MAP['EVENT'] = ['[unused12]'] 
ONTONOTES_WORD_MAP['LANGUAGE'] = ['[unused13]'] 
ONTONOTES_WORD_MAP['PERSON'] = ['[unused14]'] 
ONTONOTES_WORD_MAP['DATE'] = ['[unused15]']
ONTONOTES_WORD_MAP['MONEY'] = ['[unused16]']
ONTONOTES_WORD_MAP['LOC'] = ['[unused17]']
ONTONOTES_WORD_MAP['FAC'] = ['[unused18]']
ONTONOTES_WORD_MAP['PRODUCT'] = ['[unused19]']






# # CoNLL 03
CONLL_WORD_MAP = OrderedDict()

CONLL_WORD_MAP['O'] = ['none']
CONLL_WORD_MAP['E'] = ['[unused1]']

CONLL_WORD_MAP['ORG'] = ['[unused2]'] 
CONLL_WORD_MAP['PER'] = ['[unused3]']
CONLL_WORD_MAP['LOC'] = ['[unused4]']
CONLL_WORD_MAP['MISC'] = ['[unused5]']






# # WNUT
WNUT_WORD_MAP = OrderedDict()

WNUT_WORD_MAP['O'] = ['none']
WNUT_WORD_MAP['E'] = ['[unused1]']

WNUT_WORD_MAP['location'] = ['[unused2]']
WNUT_WORD_MAP['group'] = ['[unused3]']
WNUT_WORD_MAP['corporation'] = ['[unused4]']
WNUT_WORD_MAP['person'] = ['[unused5]']
WNUT_WORD_MAP['creative-work'] = ['[unused6]']
WNUT_WORD_MAP['product'] = ['[unused7]']






# # I2B2
I2B2_WORD_MAP = OrderedDict()

I2B2_WORD_MAP['O'] = ['none']
I2B2_WORD_MAP['E'] = ['[unused1]']

I2B2_WORD_MAP['DATE'] = ['[unused2]']
I2B2_WORD_MAP['AGE'] = ['[unused3]'] 
I2B2_WORD_MAP['STATE'] = ['[unused4]'] 
I2B2_WORD_MAP['PATIENT'] = ['[unused5]'] 
I2B2_WORD_MAP['DOCTOR'] = ['[unused6]']
I2B2_WORD_MAP['MEDICALRECORD'] = ['[unused7]']
I2B2_WORD_MAP['HOSPITAL'] = ['[unused8]']
I2B2_WORD_MAP['PHONE'] = ['[unused9]']
I2B2_WORD_MAP['IDNUM'] = ['[unused10]']
I2B2_WORD_MAP['USERNAME'] = ['[unused11]']
I2B2_WORD_MAP['STREET'] = ['[unused12]']
I2B2_WORD_MAP['CITY'] = ['[unused13]']
I2B2_WORD_MAP['ZIP'] = ['[unused14]']
I2B2_WORD_MAP['EMAIL'] = ['[unused15]']
I2B2_WORD_MAP['PROFESSION'] = ['[unused16]']
I2B2_WORD_MAP['COUNTRY'] = ['[unused17]']
I2B2_WORD_MAP['ORGANIZATION'] = ['[unused18]']
I2B2_WORD_MAP['FAX'] = ['[unused19]']





# # MIT-Movies
MOVIES_WORD_MAP = OrderedDict()

MOVIES_WORD_MAP['O'] =['none']
MOVIES_WORD_MAP['E'] =['[unused1]']

MOVIES_WORD_MAP['CHARACTER'] =  ['[unused2]']
MOVIES_WORD_MAP['GENRE'] = ['[unused3]']
MOVIES_WORD_MAP['TITLE'] = ['[unused4]']
MOVIES_WORD_MAP['PLOT'] = ['[unused5]']
MOVIES_WORD_MAP['RATING'] = ['[unused6]']
MOVIES_WORD_MAP['YEAR'] = ['[unused7]']

MOVIES_WORD_MAP['REVIEW'] = ['[unused8]']
MOVIES_WORD_MAP['ACTOR'] = ['[unused9]']
MOVIES_WORD_MAP['DIRECTOR'] = ['[unused10]']
MOVIES_WORD_MAP['SONG'] = ['[unused11]']
MOVIES_WORD_MAP['RATINGS_AVERAGE'] = ['[unused12]']
MOVIES_WORD_MAP['TRAILER'] = ['[unused13]']
