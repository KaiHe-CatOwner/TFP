"""
Author: Andrew
Date: 2021-12-04 18:19:39
LastEditors: Andrew
LastEditTime: 2022-04-29 17:13:59
FilePath: /COPNERNER/util/word_mapping.py
"""
from collections import OrderedDict

# # Few-NERD
WORD_MAP_DES = OrderedDict()

WORD_MAP_DES["O"] = ["None of the above"]
WORD_MAP_DES["E"] = ["An entity is something that exists as itself, as a subject or as an object, actually or potentially, concretely or abstractly, physically or not."]

WORD_MAP_DES["location-GPE"] = ["GPE is a geographical area which is associated with some sort of political structure"]
WORD_MAP_DES["location-bodiesofwater"] = ["Body of water is an accumulation of water on the surface of a planet"]
WORD_MAP_DES["location-island"] = ["An island (or isle) is an isolated piece of habitat that is surrounded by a dramatically different habitat, such as water"]
WORD_MAP_DES["location-mountain"] = ["A mountain is an elevated portion of the Earth's crust, generally with steep sides that show significant exposed bedrock"]
WORD_MAP_DES["location-park"] = ["A park is an area of natural, semi-natural or planted space set aside for human enjoyment and recreation or for the protection of wildlife or natural habitats"]
WORD_MAP_DES["location-road/railway/highway/transit"] = ["A road is a linear way for the conveyance of traffic that mostly has an improved surface for use by vehicles and pedestrians"]
WORD_MAP_DES["location-other"] = ["location"]

WORD_MAP_DES["person-actor"] = ["An actor or actress is a person who portrays a character in a performance"]
WORD_MAP_DES["person-artist/author"] = ["An artist or author is a person engaged in an activity related to creating art, or the creator or originator of any written work"]
WORD_MAP_DES["person-athlete"] = ["An athlete (also sportsman or sportswoman) is a person who competes in one or more sports that involve physical strength, speed or endurance"]
WORD_MAP_DES["person-director"] = ["A director controls a film's artistic and dramatic aspects and visualizes the screenplay while guiding the film crew and actors"]
WORD_MAP_DES["person-politician"] = ["A politician is a person active in party politics, or a person holding or seeking an elected office in government"]
WORD_MAP_DES["person-scholar"] = ["A scholar is a person who pursues academic and intellectual activities, particularly academics who apply their intellectualism into expertise in an area of study"]
WORD_MAP_DES["person-soldier"] = ["A soldier is a person who is a member of an army"]
WORD_MAP_DES["person-other"] = ["person"]

WORD_MAP_DES["organization-company"] = ["A company is a legal entity representing an association of people, whether natural, legal or a mixture of both, with a specific objective"]
WORD_MAP_DES["organization-education"] = ["A education organization is a place of facilitating learning, or the acquisition of knowledge, and personal development"]
WORD_MAP_DES["organization-government/governmentagency"] = ["A government is the system or group of people governing an organized community, generally a state"]
WORD_MAP_DES["organization-media/newspaper"] = ["A media or medium is the tool of communication and the link that connects the various elements of the communication process to each other"]
WORD_MAP_DES["organization-politicalparty"] = ["A political party is an organization that coordinates candidates to compete in a particular country's elections"]
WORD_MAP_DES["organization-religion"] = ["Religious groups (or religious organizations) are social organizations based on religious beliefs"]
WORD_MAP_DES["organization-showorganization"] = ["Specializing in performing arts and other activities of various professional arts performance groups"]
WORD_MAP_DES["organization-sportsleague"] = ["A sports league is a group of sports teams or individual athletes that compete against each other and gain points in a specific sport"]
WORD_MAP_DES["organization-sportsteam"] = ["A sports team is a group of individuals who play sports (sports player), usually team sports, on the same team"]
WORD_MAP_DES["organization-other"] = ["organization"]

WORD_MAP_DES["building-airport"] = ["An airport is an aerodrome with extended facilities, mostly for commercial air transport"]
WORD_MAP_DES["building-hospital"] = ["A hospital is a health care institution providing patient treatment with specialized health science and auxiliary healthcare staff and medical equipment"]
WORD_MAP_DES["building-hotel"] = ["A hotel is an establishment that provides paid lodging on a short-term basis"]
WORD_MAP_DES["building-library"] = ["A library is a collection of materials, books or media that are easily accessible for use and not just for display purposes"]
WORD_MAP_DES["building-restaurant"] = ["A restaurant is a business that prepares and serves food and drinks to customers"]
WORD_MAP_DES["building-sportsfacility"] = ["A sports facility is any location where sports is provided"]
WORD_MAP_DES["building-theater"] = ["Theatre or theater is a collaborative form of performing art that uses live performers"]
WORD_MAP_DES["building-other"] = ["building"]

WORD_MAP_DES["art-broadcastprogram"] = ["Broadcasting is the distribution of audio or video content to a dispersed audience via any electronic mass communications medium"]
WORD_MAP_DES["art-film"] = ["A film also called a movie, is a work of visual art that simulates experiences through the use of moving images"]
WORD_MAP_DES["art-music"] = ["Music is the art of arranging sounds in time through the elements of melody, harmony, rhythm, and timbre"]
WORD_MAP_DES["art-painting"] = ["Painting is the practice of applying paint, pigment, color or other medium to a solid surface"]
WORD_MAP_DES["art-writtenart"] = ["Writing is a medium of human communication that involves the representation of a language through symbols"]
WORD_MAP_DES["art-other"] = ["art"]

WORD_MAP_DES["product-airplane"] = ["An airplane or aeroplane (informally plane) is a fixed-wing aircraft that is propelled forward by thrust from a jet engine, propeller, or rocket engine"]
WORD_MAP_DES["product-car"] = ["A car (or automobile) is a wheeled motor vehicle used for transportation"]
WORD_MAP_DES["product-food"] = ["Food is any substance consumed to provide nutritional support for an organism"]
WORD_MAP_DES["product-game"] = ["A game is a structured form of play, usually undertaken for entertainment or fun, and sometimes used as an educational tool"]
WORD_MAP_DES["product-ship"] = ["A ship is a large watercraft that travels the world's oceans and other sufficiently deep waterways, carrying cargo or passengers"]
WORD_MAP_DES["product-software"] = ["Software is a collection of instructions that tell a computer how to work"]
WORD_MAP_DES["product-train"] = ["A train is a series of connected vehicles that run along a railway track and transport people or freight"]
WORD_MAP_DES["product-weapon"] = ["A weapon, arm or armament is any implement or device that can be used with the intent to inflict physical damage or harm"]
WORD_MAP_DES["product-other"] = ["product"]

WORD_MAP_DES["event-attack/battle/war/militaryconflict"] = ["War is an intense armed conflict between states, governments, societies, or paramilitary groups such as mercenaries, insurgents, and militias"]
WORD_MAP_DES["event-disaster"] = ["A disaster is a serious problem occurring over a short or long period of time that causes widespread human, material, economic or environmental loss"]
WORD_MAP_DES["event-election"] = ["An election is a formal group decision-making process by which a population chooses an individual or multiple individuals to hold public office"]
WORD_MAP_DES["event-protest"] = ["A protest (demonstration, remonstration or remonstrance) is a public expression of objection, disapproval or dissent towards an idea or action, typically a political one"]
WORD_MAP_DES["event-sportsevent"] = ["Sport event is any form of competitive physical activity or game"]
WORD_MAP_DES["event-other"] = ["event"]

WORD_MAP_DES["other-astronomything"] = ["Astronomy is a natural science that studies celestial objects and phenomena"]
WORD_MAP_DES["other-award"] = ["An award is something given to a recipient as a token of recognition of excellence in a certain field"]
WORD_MAP_DES["other-biologything"] = ["Biology is the scientific study of life"]
WORD_MAP_DES["other-chemicalthing"] = ["Chemistry is the scientific study of the properties and behavior of matter"]
WORD_MAP_DES["other-currency"] = ["A currency is a standardization of money in any form when in use or circulation as a medium of exchange, for example banknotes and coins"]
WORD_MAP_DES["other-disease"] = ["A disease is a particular abnormal condition that negatively affects the structure or function of all or part of an organism"]
WORD_MAP_DES["other-educationaldegree"] = ["An educational degree is a qualification awarded to students upon successful completion of a course of study in higher education, usually at a college or university"]
WORD_MAP_DES["other-god"] = ["God is usually viewed as the supreme being, creator, and principal object of faith"]
WORD_MAP_DES["other-language"] = ["A language is a structured system of communication"]
WORD_MAP_DES["other-law"] = ["Law is a system of rules created and enforced through social or governmental institutions to regulate behavior"]
WORD_MAP_DES["other-livingthing"] = ["Living thing is organism, is any organic living system that operates"]
WORD_MAP_DES["other-medical"] = ["medical is the science and practice of caring for a patient, managing the diagnosis, palliation of their injury or disease, and promoting their health"]


# # OntoNotes
ONTONOTES_WORD_MAP_DES = OrderedDict()

ONTONOTES_WORD_MAP_DES["O"] = ["None of the above"]
ONTONOTES_WORD_MAP_DES["E"] = ["An entity is something that exists as itself, as a subject or as an object, actually or potentially, concretely or abstractly, physically or not."]

ONTONOTES_WORD_MAP_DES["ORG"] = ["An organization, or organisation, is an entity—such as a company, an institution, or an association—comprising one or more people and having a particular purpose"]
ONTONOTES_WORD_MAP_DES["NORP"] = ["A country or state (sometimes called nation) is a distinct territorial body or political entity"]
ONTONOTES_WORD_MAP_DES["ORDINAL"] = ["Ordinal numerals or ordinal number words are words representing position or rank in a sequential order"]
ONTONOTES_WORD_MAP_DES["WORK_OF_ART"] = ["Art is a diverse range of human activity, and resulting product, that involves creative or imaginative talent expressive of technical proficiency, beauty, emotional power, or conceptual ideas"]
ONTONOTES_WORD_MAP_DES["QUANTITY"] = ["Quantity or amount is a property that can exist as a multitude or magnitude, which illustrate discontinuity and continuity"]
ONTONOTES_WORD_MAP_DES["LAW"] = ["Law is a system of rules created and enforced through social or governmental institutions to regulate behavior"]
ONTONOTES_WORD_MAP_DES["GPE"] = ["A nation is a community of people formed on the basis of a combination of shared features such as language, history, ethnicity, culture and/or territory"]
ONTONOTES_WORD_MAP_DES["CARDINAL"] = ["Cardinal numeral (or cardinal number word) is a part of speech used to count"]
ONTONOTES_WORD_MAP_DES["PERCENT"] = ["A percentage is a number or ratio expressed as a fraction of 100"]
ONTONOTES_WORD_MAP_DES["TIME"] = ["Time is the continued sequence of existence and events that occurs in an apparently irreversible succession from the past, through the present, into the future"]
ONTONOTES_WORD_MAP_DES["EVENT"] = ["An event is something that happens"]   # '#####'
ONTONOTES_WORD_MAP_DES["LANGUAGE"] = ["A language is a structured system of communication"]
ONTONOTES_WORD_MAP_DES["PERSON"] = ["A person is a being that has certain capacities or attributes such as reason, morality, consciousness or self-consciousness, and being a part of a culturally established form of social relations such as kinship, ownership of property, or legal responsibility"]
ONTONOTES_WORD_MAP_DES["DATE"] = ["A date is a reference to a particular day represented within a calendar system"]
ONTONOTES_WORD_MAP_DES["MONEY"] = ["Money is any item or verifiable record that is generally accepted as payment for goods and services and repayment of debts, such as taxes, in a particular country or socio-economic context"]
ONTONOTES_WORD_MAP_DES["LOC"] = ["Location or place are used to denote a region (point, line, or area) on Earth's surface or elsewhere"]
ONTONOTES_WORD_MAP_DES["FAC"] = ["A facility is a commercial or institutional building, such as a hotel, resort, school, office complex, sports arena, or convention center"]
ONTONOTES_WORD_MAP_DES["PRODUCT"] = ["A product is an object, or system, or service made available for consumer use as of the consumer demand"]






# # # OntoNotes
# ONTONOTES_11_WORD_MAP_DES = OrderedDict()

# ONTONOTES_11_WORD_MAP_DES["O"] = ["None of the above"]
# # ONTONOTES_WORD_MAP_DES["E"] = ["An entity is something that exists as itself, as a subject or as an object, actually or potentially, concretely or abstractly, physically or not."]

# ONTONOTES_11_WORD_MAP_DES["WORK_OF_ART"] = ["Art is a diverse range of human activity, and resulting product, that involves creative or imaginative talent expressive of technical proficiency, beauty, emotional power, or conceptual ideas"]
# ONTONOTES_11_WORD_MAP_DES["ORG"] = ["An organization, or organisation, is an entity—such as a company, an institution, or an association—comprising one or more people and having a particular purpose"]
# ONTONOTES_11_WORD_MAP_DES["NORP"] = ["A country or state (sometimes called nation) is a distinct territorial body or political entity"]
# ONTONOTES_11_WORD_MAP_DES["LANGUAGE"] = ["A language is a structured system of communication"]
# ONTONOTES_11_WORD_MAP_DES["PERSON"] = ["A person is a being that has certain capacities or attributes such as reason, morality, consciousness or self-consciousness, and being a part of a culturally established form of social relations such as kinship, ownership of property, or legal responsibility"]
# ONTONOTES_11_WORD_MAP_DES["LAW"] = ["Law is a system of rules created and enforced through social or governmental institutions to regulate behavior"]
# ONTONOTES_11_WORD_MAP_DES["EVENT"] = ["An event is something that happens"]   # '#####'
# ONTONOTES_11_WORD_MAP_DES["FAC"] = ["A facility is a commercial or institutional building, such as a hotel, resort, school, office complex, sports arena, or convention center"]
# ONTONOTES_11_WORD_MAP_DES["LOC"] = ["Location or place are used to denote a region (point, line, or area) on Earth's surface or elsewhere"]
# ONTONOTES_11_WORD_MAP_DES["GPE"] = ["A nation is a community of people formed on the basis of a combination of shared features such as language, history, ethnicity, culture and/or territory"]
# ONTONOTES_11_WORD_MAP_DES["PRODUCT"] = ["A product is an object, or system, or service made available for consumer use as of the consumer demand"]











# # CoNLL 03
CONLL_WORD_MAP_DES = OrderedDict()

CONLL_WORD_MAP_DES["O"] = ["None of the above"]
CONLL_WORD_MAP_DES["E"] = ["An entity is something that exists as itself, as a subject or as an object, actually or potentially, concretely or abstractly, physically or not."]

CONLL_WORD_MAP_DES["ORG"] = ["An organization, or organisation, is an entity—such as a company, an institution, or an association—comprising one or more people and having a particular purpose"]
CONLL_WORD_MAP_DES["PER"] = ["A person is a being that has certain capacities or attributes such as reason, morality, consciousness or self-consciousness, and being a part of a culturally established form of social relations such as kinship, ownership of property, or legal responsibility"]
CONLL_WORD_MAP_DES["LOC"] = ["Location or place are used to denote a region (point, line, or area) on Earth's surface or elsewhere"]
CONLL_WORD_MAP_DES["MISC"] = ["other"]










# # WNUT
WNUT_WORD_MAP_DES = OrderedDict()

WNUT_WORD_MAP_DES["O"] = ["None of the above"]
WNUT_WORD_MAP_DES["E"] = ["An entity is something that exists as itself, as a subject or as an object, actually or potentially, concretely or abstractly, physically or not."]

WNUT_WORD_MAP_DES["location"] = ["Location or place are used to denote a region (point, line, or area) on Earth's surface or elsewhere"]
WNUT_WORD_MAP_DES["group"] = ["A group is a number of people or things that are located, gathered, or classed together"]
WNUT_WORD_MAP_DES["corporation"] = ["A corporation is an organization—usually a group of people or a company—authorized by the state to act as a single entity and recognized as such in law for certain purposes"]
WNUT_WORD_MAP_DES["person"] = ["A person is a being that has certain capacities or attributes such as reason, morality, consciousness or self-consciousness, and being a part of a culturally established form of social relations such as kinship, ownership of property, or legal responsibility"]
WNUT_WORD_MAP_DES["creative-work"] = ["Creativity is a phenomenon whereby something new and valuable is formed"] # '#####'
WNUT_WORD_MAP_DES["product"] = ["A product is an object, or system, or service made available for consumer use as of the consumer demand"]











# # I2B2
I2B2_WORD_MAP_DES = OrderedDict()

I2B2_WORD_MAP_DES["O"] = ["None of the above"]
I2B2_WORD_MAP_DES["E"] = ["An entity is something that exists as itself, as a subject or as an object, actually or potentially, concretely or abstractly, physically or not."]

I2B2_WORD_MAP_DES["DATE"] = ["A date is a reference to a particular day represented within a calendar system"] #
I2B2_WORD_MAP_DES["AGE"] = ["Age is the amount of time something has been alive or has existed"] #
I2B2_WORD_MAP_DES["STATE"] = ["A state is a territorial and constitutional community forming part of a federation"] #
I2B2_WORD_MAP_DES["PATIENT"] = ["A patient is any recipient of health care services that are performed by healthcare professionals"]
I2B2_WORD_MAP_DES["DOCTOR"] = ["A doctor is a medical practitioner"]
I2B2_WORD_MAP_DES["MEDICALRECORD"] = ["Medical record is the systematic documentation of a single patient's medical history and care across time within one particular health care provider's jurisdiction"]  # '#####'
I2B2_WORD_MAP_DES["HOSPITAL"] = ["A hospital is a health care institution providing patient treatment with specialized health science and auxiliary healthcare staff and medical equipment"]
I2B2_WORD_MAP_DES["PHONE"] = ["A number assigned to a telephone line for a specific phone or set of phones (as for a residence) that is used to call that phone"]
I2B2_WORD_MAP_DES["IDNUM"] = ["An ID number is a sequence of numbers that identifies an object"] # '#####'
I2B2_WORD_MAP_DES["USERNAME"] = ["Username is a sequence of characters that identifies a user"]
I2B2_WORD_MAP_DES["STREET"] = ["A street is a public thoroughfare in a built environment"]
I2B2_WORD_MAP_DES["CITY"] = ["A city is a large human settlement"]
I2B2_WORD_MAP_DES["ZIP"] = ["A zip is a number that identifies a particular postal delivery area in the U.S"] # '#####'
I2B2_WORD_MAP_DES["EMAIL"] = ["Email is a method of exchanging messages between people using electronic devices"]
I2B2_WORD_MAP_DES["PROFESSION"] = ["A profession is a disciplined group of individuals who adhere to ethical standards and who hold themselves out as, and are accepted by the public as possessing special knowledge and skills in a widely recognised body of learning derived from research, education and training at a high level, and who are prepared to apply this knowledge and exercise these skills in the interest of others."]
I2B2_WORD_MAP_DES["COUNTRY"] = ["A country or state (sometimes called nation) is a distinct territorial body or political entity"]
I2B2_WORD_MAP_DES["ORGANIZATION"] = ["An organization, or organisation, is an entity—such as a company, an institution, or an association—comprising one or more people and having a particular purpose"]
I2B2_WORD_MAP_DES["FAX"] = ["A fax is a device used to send or receive facsimile communications"]  # '#####'

# I2B2_WORD_MAP_DES["LOCATION-OTHER"] = ["location"] 
# I2B2_WORD_MAP_DES["DEVICE"] = ["A device is usually a constructed tool"] 
# I2B2_WORD_MAP_DES["BIOID"] = ["A bio is a biography or biographical sketch"]  # '#####'
# I2B2_WORD_MAP_DES["HEALTHPLAN"] = ["A plan is typically any diagram or list of steps with details of timing and resources, used to achieve an objective to do something"] # '#####'
# I2B2_WORD_MAP_DES["URL"] = ["A url is a reference to a web resource that specifies its location on a computer network and a mechanism for retrieving it"]












# # MIT-Movies
MOVIES_WORD_MAP_DES = OrderedDict()

MOVIES_WORD_MAP_DES["O"] = ["none of the above"]
MOVIES_WORD_MAP_DES["E"] = ["An entity is something that exists as itself, as a subject or as an object, actually or potentially, concretely or abstractly, physically or not."]

MOVIES_WORD_MAP_DES["CHARACTER"] = ["In fiction, a character (or speaker, in poetry) is a person or other being in a narrative (such as a novel, play, radio or television series, music, film, or video game)"]
MOVIES_WORD_MAP_DES["GENRE"] = ["Genre is any form or type of communication in any mode (written, spoken, digital, artistic, etc.) with socially-agreed-upon conventions developed over time"]
MOVIES_WORD_MAP_DES["TITLE"] = ["Title is the distinguishing name of a written, printed, or filmed production"]
MOVIES_WORD_MAP_DES["PLOT"] = ["In a literary work, film, or other narrative, the plot is the sequence of events where each affects the next one through the principle of cause-and-effect"]
MOVIES_WORD_MAP_DES["RATING"] = ["A rating is an evaluation or assessment of something, in terms of quality, quantity, or some combination of both"]
MOVIES_WORD_MAP_DES["YEAR"] = ["A calendar year specified usually by a number"]

MOVIES_WORD_MAP_DES["REVIEW"] =["A review is an evaluation of a publication, product, service, or company or a critical take on current affairs in literature, politics or culture"]   # '#####'
MOVIES_WORD_MAP_DES["ACTOR"] = ["An actor or actress is a person who portrays a character in a performance"]
MOVIES_WORD_MAP_DES["DIRECTOR"] = ["A director controls a film's artistic and dramatic aspects and visualizes the screenplay while guiding the film crew and actors"]
MOVIES_WORD_MAP_DES["SONG"] = ["A song is a musical composition intended to be performed by the human voice"]
MOVIES_WORD_MAP_DES["RATINGS_AVERAGE"] = ["An average is a single number taken as representative of a list of numbers, usually the sum of the numbers divided by how many numbers are in the list (the arithmetic mean)"]
MOVIES_WORD_MAP_DES["TRAILER"] = ["Trailer (promotion), an advertisement, usually in the form of a brief excerpt or string of excerpts, for a media work"]