import random
from collections import defaultdict

from typing import Union, List


class RandomHardPromptVerbalizer:
    def __init__(self):
        def_kimi = {
            'Accident': 'An unforeseen and unplanned event that results in damage or injury, often involving a sudden visual change or disruption.',
            'AirAccident': 'An accident involving an aircraft, characterized by a sudden and unexpected change in its flight path or a visible crash.',
            'AnimalAttackAnimal': 'A violent confrontation between two animals, often involving visible physical aggression and damage to the animals involved.',
            'AnimalAttackHuman': 'An incident where an animal physically attacks a human, resulting in visible harm or injury to the person.',
            'AnimalPredation': 'The act of an animal hunting and killing another for food, which can be observed through the visual display of hunting behaviors and the aftermath of the kill.',
            'CarAccident': 'A collision or crash involving one or more vehicles, typically resulting in visible damage to the cars and potential injuries to occupants.',
            'Collapse': 'The sudden falling down of a structure or object, which can be seen as a dramatic change in the visual appearance of the object.',
            'CrowdViolence': 'A chaotic and aggressive situation involving a group of people, often characterized by visible physical altercations and property damage.',
            'Explosion': 'A sudden and powerful release of energy, resulting in a visible flash, smoke, and debris, often causing damage to the surrounding area.',
            'FallDown': 'The act of an object or person falling to the ground, which can be visually observed as a rapid change in position.',
            'FallIntoWater': 'The act of an object or person falling into a body of water, which can be visually identified by the splash and subsequent disappearance of the object.',
            'Fire': 'A visible, self-sustaining chemical reaction that releases heat and light, often causing damage to materials it comes into contact with.',
            'MechanicalAccident': 'An unexpected event at an industrial site, often involving visible damage to machinery or structures, and potential harm to workers.',
            'ObjectImpact': 'The visible collision of an object with another object or surface, resulting in a change in the state or position of the impacted object.',
            'RangeShooting': 'The act of firing a weapon at a target within a designated area, which can be observed through the visual trajectory of the projectiles.',
            'Riot': 'A large-scale, public disorder involving violence and destruction, characterized by visible chaos and damage to property.',
            'Robbery': 'The act of stealing from a person or place using force or threat of force, which may involve visible confrontations and property damage.',
            'Shooting': 'The act of discharging a firearm, which can be visually identified by the muzzle flash and the trajectory of the bullets.',
            'TrainAccident': 'An incident involving a train, such as a derailment or collision, resulting in visible damage to the train and potential injuries.',
            'Violence': 'The use of physical force intended to hurt, damage, or kill someone or something, which can be observed through visible aggression and its aftermath.',
            'WarScene': 'A scene depicting armed conflict, often characterized by visible destruction, combatants, and the aftermath of battles.',
            "Abuse": "The act of causing physical harm or suffering to an animal, often involving visible signs of injury or distress.",
            "Arrest": "The act of taking someone into custody by law enforcement, typically involving handcuffs and visible law enforcement presence.",
            "Arson": "The intentional setting of fire to property, resulting in visible damage or destruction, often with flames and smoke.",
            "Assault": "The act of physically attacking someone without consent, which may involve visible signs of injury on the victim.",
            "Burglary": "The act of illegally entering a building with the intent to commit a crime, often characterized by broken locks or windows and missing items.",
            "Shoplifting": "The act of stealing merchandise from a store without paying, often involving hidden merchandise and attempts to avoid detection by store employees.",
            "Stealing": "The act of taking someone else's property without permission, which may involve the concealment of the stolen items and a lack of visible transaction.",
            "Vandalism": "The act of deliberately damaging property, often resulting in visible graffiti, broken windows, or other forms of property damage."
        }

        def_ernie = {
            'Accident': "An unexpected event captured visually, often involving damage or injury.",
            'AirAccident': "A visual incident involving an aircraft, such as a crash or collision.",
            'AnimalAttackAnimal': "A visual scene where one animal attacks another.",
            'AnimalAttackHuman': "A depiction of an animal attacking a human being.",
            'AnimalPredation': "A visual representation of an animal hunting and killing another animal for food.",
            'CarAccident': "A visual collision or mishap involving one or more cars.",
            'Collapse': "A visual event where a structure, such as a building, falls down or caves in.",
            'CrowdViolence': "A scene of violent behavior involving a large group of people.",
            'Explosion': "A sudden, violent release of energy, often captured visually with fire and debris.",
            'FallDown': "A visual depiction of someone or something falling to the ground.",
            'FallIntoWater': "A scene where a person or object falls into a body of water.",
            'Fire': "A visual display of burning, often with flames and smoke.",
            'MechanicalAccident': "A visual incident occurring in a factory setting, often involving machinery or chemicals.",
            'ObjectImpact': "A visual representation of an object striking another object with force.",
            'RangeShooting': "A visual scene of shooting targets at a firing range.",
            'Riot': "A visual depiction of a violent disturbance involving a group of people.",
            'Robbery': "A visual representation of a theft involving force or the threat of force.",
            'Shooting': "A visual scene of discharging a firearm or other weapon.",
            'TrainAccident': "A visual incident involving a train, such as a derailment or collision.",
            'Violence': "A visual representation of aggressive, harmful behavior intended to cause injury or pain.",
            'WarScene': "A visual depiction of combat or military conflict, often with destruction and chaos.",
            "Abuse": "An event where an animal is harmed, often visible through signs of injury or distress.",
            "Arrest": "A situation where a person is taken into custody by authorities, usually involving handcuffs and police presence.",
            "Arson": "An incident of deliberate fire-setting, characterized by flames and smoke emanating from a building or object.",
            "Assault": "A violent attack on a person, often visible through physical altercations, injuries, or the presence of force.",
            "Burglary": "An unauthorized entry into a building, typically noticeable by forced open doors, broken windows, or missing items.",
            "Shoplifting": "The act of stealing merchandise from a store, often caught on surveillance cameras showing someone concealing items and leaving without paying.",
            "Stealing": "The unauthorized taking of someone else's property, often evident through missing items and surveillance footage of the theft.",
            "Vandalism": "The deliberate destruction or damage to property, noticeable through graffiti, broken windows, or other forms of physical alteration."
        }

        def_spark = {
            'Accident': 'An unexpected event that causes injury or damage.',
            'AirAccident': 'An accident involving aircraft, often resulting in crashes or mishaps.',
            'AnimalAttackAnimal': 'A situation where one animal attacks another animal.',
            'AnimalAttackHuman': 'A situation where an animal attacks a human being.',
            'AnimalPredation': 'The act of animals hunting and eating other animals.',
            'CarAccident': 'An accident involving cars, which can result in injuries or property damage.',
            'Collapse': 'The sudden failure or breakdown of something, often visually dramatic.',
            'CrowdViolence': 'Violent behavior by a group of people, often leading to conflict or chaos.',
            'Explosion': 'A loud noise and flash caused by something exploding, often with destruction.',
            'FallDown': 'The act of falling down, which can be due to various reasons like tripping or losing balance.',
            'FallIntoWater': 'The act of falling into water, which can be dangerous and cause drowning.',
            'Fire': 'A burning material that can cause damage or harm.',
            'MechanicalAccident': 'An accident that occurs in a factory setting, potentially causing harm to workers and equipment.',
            'ObjectImpact': 'The collision between objects, which can cause damage or injury.',
            'RangeShooting': 'The act of shooting at a target from a distance, often used in sports or military contexts.',
            'Riot': 'A large-scale public disturbance involving violence and chaotic behavior.',
            'Robbery': 'The act of stealing from someone, often committed by a group of people.',
            'Shooting': 'The act of firing a weapon, which can be lethal if aimed at someone.',
            'TrainAccident': 'An accident involving trains, which can result in derailments or collisions.',
            'Violence': 'The use of physical force to cause harm or injury.',
            'WarScene': 'A scene depicting warfare, involving battles, combat, and destruction.',
            "Abuse": "The act of harming or mistreating an animal, often involving physical violence.",
            "Arrest": "The act of taking someone into custody by the police for investigation or legal action.",
            "Arson": "The deliberate burning of property to cause damage or destroy evidence.",
            "Assault": "The act of attacking someone with the intent to cause them physical harm or fear.",
            "Burglary": "The unlawful entry of a structure to commit a crime, typically involving theft.",
            "Shoplifting": "The act of stealing goods from a store without paying for them, usually done by customers.",
            "Stealing": "The act of taking something that belongs to someone else without permission, intending to deprive the owner of it.",
            "Vandalism": "The deliberate destruction or damaging of public or private property for no other reason than pleasure or to cause trouble."
        }

        def_qwen = {
            'Accident': 'An unexpected event, often resulting in damage or injury, involving vehicles, machinery, or other objects.',
            'AirAccident': 'A crash or serious malfunction of an aircraft, typically involving structural failure, engine problems, or pilot error, leading to debris and possible casualties.',
            'AnimalAttackAnimal': 'A confrontation between two or more animals where one attempts to harm the other, often seen with teeth, claws, and aggressive body postures.',
            'AnimalAttackHuman': 'An incident where an animal aggressively assaults a human, which may involve biting, scratching, or charging, often occurring in natural habitats or domestic settings.',
            'AnimalPredation': 'The act of an animal hunting and consuming another for food, characterized by stalking, chasing, and killing prey, usually with specific hunting techniques.',
            'CarAccident': 'A collision involving one or more vehicles, which can result in damaged cars, injured passengers, and sometimes road debris or skid marks on the ground.',
            'Collapse': 'The sudden failure and falling down of a structure, such as a building or bridge, which can create a pile of rubble and dust clouds.',
            'CrowdViolence': 'A chaotic situation in a group of people that turns violent, marked by pushing, punching, kicking, and the use of makeshift weapons.',
            'Explosion': 'A rapid release of energy causing a loud noise and a visible shockwave, often resulting in fire, smoke, and scattered debris.',
            'FallDown': 'The action of someone losing balance and dropping to the ground, which can happen due to tripping, slipping, or being pushed.',
            'FallIntoWater': 'An unexpected descent into a body of water, such as a river or pool, which can be accidental or deliberate but is often accompanied by splashing and flailing arms.',
            'Fire': 'A chemical reaction that releases heat and light, typically involving flames, smoke, and the destruction of materials, which can spread rapidly if not controlled.',
            'MechanicalAccident': 'An incident in a manufacturing plant that results in injuries, fatalities, or environmental damage, often due to machinery malfunctions or hazardous material leaks.',
            'ObjectImpact': 'The moment when an object strikes another with force, potentially causing damage or injury, visible through dents, cracks, or broken pieces.',
            'RangeShooting': 'The practice of firing guns at a shooting range, where individuals aim at targets from a set distance, often seen with the sound of gunfire and the sight of bullet holes in targets.',
            'Riot': 'A violent public disorder where a crowd acts together in a destructive manner, often involving breaking windows, setting fires, and clashing with law enforcement.',
            'Robbery': 'The act of stealing from a person or place by using force or threats, which can include the perpetrator wearing a mask, wielding a weapon, and fleeing with stolen goods.',
            'Shooting': "The act of discharging a firearm, which can be heard as a loud bang and seen as a flash from the gun barrel, followed by the bullet's trajectory.",
            'TrainAccident': 'A collision or derailment involving a train, which can lead to twisted metal, overturned carriages, and debris scattered along the tracks.',
            'Violence': 'Physical force exerted for the purpose of harming or injuring others, which can be seen in punches, kicks, and the use of weapons in fights or assaults.',
            'WarScene': 'A combat scenario during armed conflict, featuring explosions, gunfire, smoke, and the movement of military vehicles and personnel.',
            "Abuse": "An act where an animal is being mistreated, often visible by signs of physical harm or distress to the animal.",
            "Arrest": "The process of a law enforcement officer taking a person into custody, typically involving handcuffs and the individual being placed in a police vehicle.",
            "Arson": "The deliberate setting of a fire, usually characterized by flames, smoke, and potential destruction of property or natural elements.",
            "Assault": "A violent physical attack on a person, observable through aggressive actions such as hitting, pushing, or threatening with a weapon.",
            "Burglary": "The unlawful entry into a building or structure, often with the intent to commit a theft, which may be indicated by broken windows, forced doors, or items in disarray.",
            "Shoplifting": "The theft of goods from a store without paying, typically observed when someone conceals merchandise and tries to leave the premises without going through checkout.",
            "Stealing": "Taking someone else's property without permission, often seen as a person carrying away items that do not belong to them, possibly looking around furtively.",
            "Vandalism": "The deliberate destruction or damage to property, which can be recognized by graffiti, broken objects, or other forms of defacement."
        }

        def_gemini = {
            "Accident": "An unexpected and unfortunate event, often resulting in injury or damage.",
            "AirAccident": "An accident involving an aircraft, such as a plane crash or mid-air collision.",
            "AnimalAttackAnimal": "An aggressive encounter between two or more animals, often resulting in injury or death.",
            "AnimalAttackHuman": "An aggressive encounter between an animal and a human, often resulting in injury or death.",
            "AnimalPredation": "The act of one animal hunting and killing another for food.",
            "CarAccident": "A collision between two or more vehicles, often resulting in injury or damage.",
            "Collapse": "The sudden and unexpected falling down or caving in of a structure.",
            "CrowdViolence": "Violent behavior by a group of people, often resulting in injury or property damage.",
            "Explosion": "A sudden and violent release of energy, often accompanied by a loud noise and blast wave.",
            "FallDown": "The act of falling to the ground, often due to loss of balance or injury.",
            "FallIntoWater": "The act of falling into a body of water, often resulting in drowning or injury.",
            "Fire": "A destructive process involving the rapid oxidation of a combustible material, releasing heat, light, and smoke.",
            "FactoryAccident": "An accident that occurs in a factory or industrial setting, often resulting in injury or damage.",
            "ObjectImpact": "The collision of two or more objects, often resulting in damage or destruction.",
            "RangeShooting": "The practice of shooting at targets at a designated range, often for sport or training.",
            "Riot": "A violent public disturbance, often involving a large group of people.",
            "Robbery": "The act of taking property from another person by force or threat of force.",
            "Shooting": "The act of firing a firearm, often with the intent to harm or kill someone or something.",
            "TrainAccident": "An accident involving a train, such as a derailment or collision.",
            "Violence": "The use of physical force or threats to harm or intimidate others.",
            "WarScene": "A situation or event related to armed conflict, often involving fighting, destruction, and casualties.",
            "Abuse": "Causing intentional harm or suffering to an animal, such as physical abuse, neglect, or cruelty.",
            "Arrest": "The act of taking a person into custody by law enforcement officials, usually based on suspicion of a crime.",
            "Arson": "The intentional and unlawful setting of fire to property, often with the intent to cause damage or destruction.",
            "Assault": "A violent act or threat of violence against another person, including physical or verbal attacks.",
            "Burglary": "The unlawful breaking and entering into a building with the intent to commit a felony, such as theft or assault.",
            "Shoplifting": "The theft of merchandise from a retail store, usually by concealing items and leaving without paying.",
            "Stealing": "The act of taking property that belongs to another person without permission, often with the intent to keep it permanently.",
            "Vandalism": "The intentional destruction or damage of property, often in a public or private space, for malicious reasons."
        }

        # aggregate def from all sources
        cls2text = defaultdict(list)
        for k, v in def_kimi.items():
            cls2text[k].append(f"{k}, {v}")
        for k, v in def_qwen.items():
            cls2text[k].append(f"{k}, {v}")
        for k, v in def_ernie.items():
            cls2text[k].append(f"{k}, {v}")
        for k, v in def_spark.items():
            cls2text[k].append(f"{k}, {v}")
        for k, v in def_gemini.items():
            cls2text[k].append(f"{k}, {v}")
        # add vanilla word
        for k in cls2text.keys():
            cls2text[k].append(k)
        cls2text['Normal'].append('Normal')
        cls2text['Normal'].append('Normal. Normal behavior, views or surveillance content captured by camera that not involve any unexpected or unusual events.')
        # add class alias
        cls2text['RoadAccidents'] = cls2text['CarAccident']
        cls2text['Car accident'] = cls2text['CarAccident']
        cls2text['Fighting'] = cls2text['Violence']

        self.cls2text = cls2text

    def __call__(self, inputs: Union[str, List[str]]):
        if type(inputs) is str:
            return random.choice(self.cls2text[inputs])
        elif type(inputs) is list:
            return [random.choice(self.cls2text[i]) for i in inputs]


class DatasetSpecVerbalizer:
    def __init__(self):
        universal_cls_defs = {
            'Normal': [
                'Normal',
                'Normal user uploaded video',
                'Normal surveillance footage.',
                'Normal behavior, views or surveillance content captured by camera that not involve any unexpected or unusual events.',
            ],
            'Assault': [
                'Assault',
                'The man unilaterally attacked the victim, using fists or sticks',
                'Multiple people surround and assault one person with fists and cudgels.',
                'Continuous assault by one person on another',
            ],
            'AirAccident': [
                'Airplane crash scene',
                'Air accident',
                'The plane crashed from the sky, emitting black smoke',
                'The helicopter is hovering and falling to the ground, there may be an explosion',
            ],
            'AnimalAttackAnimal': [
                'An animal attacks another animal, often resulting in injury or death.',
                'Animals attack each other',
            ],
            'AnimalAttackHuman': [
                'An animal attacks a person, often resulting in injury or death.',
                'Animals (such as cows, bears, and deer) attack people and collide with property.',
                'Mad dog barks at people and bites them',
            ],
            'AnimalPredation': [
                'Predation',
                'Predatory scenes, often bloody and violent',
                'Animal hunting and consuming another for food, characterized by stalking, chasing, and killing prey.',
                'Animal hunting and killing another animal for food',
            ],
            'CarAccident': [
                'A collision between two or more vehicles, often resulting in injury or damage.',
                'Traffic accident scene, vehicle colliding with pedestrian',
                'The driving recorder recorded that two cars collided with each other',
                'Road accident scene',
            ],
            'Collapse': [
                'Collapse scene',
                'The sudden falling down of a structure, such as a building or bridge, creating a pile of rubble and dust clouds.',
                'A visual event where a structure, such as a building, falls down or caves in.',
            ],
            'CrowdViolence': [
                'A Crowd violence scene',
                'A chaotic situation in a group of people that turns violent, marked by pushing, punching, kicking.',
                'Violent behavior by a group of people, often leading to conflict or chaos.',
                "A scene of violent behavior involving a large group of people.",
            ],
            'Explosion': [
                'Explosion',
                'Explosion, often resulting in fire, smoke, and scattered debris.',
                'The scene of the explosion, with mushroom clouds and smoke after the explosion.',
            ],
            'FallDown': [
                'People fall down',
                'Someone losing balance and dropping to the ground, which can happen due to tripping, slipping, or being pushed.',
                'Falling while running',
                'People falling from a height',
            ],
            'Fighting': [
                'Fighting',
                'Violence',
                'Using violence to injure or kill someone, usually involving group fights.',
                'A group of people fighting and brawling, which can be seen in punches, kicks.',
                'In sports, players have conflicts and start fighting each other.',
                'A man knocked down another person to the ground.'
            ],
            'Fire': [
                'Fire disaster',
                'Fire accident. Burning properties and thick black smoke can be seen',
                'Flames burn in unexpected places',
            ],
            'FallIntoWater': [
                'Someone falling into water',
                'A scene where a person or object falls into a body of water.',
                'An object or person falling into water, often involving splash and subsequent disappearance of the object.',
            ],
            'MechanicalAccident': [
                'Accident due to machinery failures, often happened in factories, warehouses, or other industrial settings.',
                'An incident in a manufacturing plant that results in injuries, fatalities, or environmental damage, often due to machinery malfunctions or hazardous material leaks.',
                'Accident happened in a industrial setting.'
            ],
            'ObjectImpact': [
                'Object falls down',
                'An object strikes another, visible through dents, cracks, or broken pieces.',
                'An object falls and potentially hits a person',
            ],
            'Riot': [
                'Riot scene',
                'The chaotic riot scene. There are many people and special police officers who are suppressing it.',
                'large-scale, public riot, often involving breaking windows, setting fires, and clashing with law enforcement.',
                'Riot scene, armed police wearing helmets and holding shields forming a human wall, with smoke and flames in the background.',
                'Riot scene. The crowd marched with flags and slogans with police holding shields to form a human wall suppressing the riot. ',
            ],
            'Robbery': [
                'Robbery',
                'Robbing others property through violent means such as beating or holding a gun',
            ],
            'Shooting': [
                'Shooting',
                'Firing a weapon, usually involving muzzle flash and the trajectory of the bullets',
                'The act of firing a firearm, often with gun flame and people lying down.',
                'A person points a gun at another person and shoots, and the muzzle emits flames and smoke.'
            ],
            'TrainAccident': [
                'Train accident',
                'A collision or derailment involving a train',
                'An accident involving trains, which can result in derailments or collisions.',
                'The train collided with vehicles at the gate',
            ],
            'WarScene': [
                'War scene',
                'War scene, often involving gunfire, explosions, cannon, tanks, and blood.',
                'War scene, with fully armed soldiers carrying out tasks',
                'A combat scenario during armed conflict, featuring explosions, gunfire, smoke, and the movement of military vehicles.',
                'A scene depicting warfare, involving battles, combat, and destruction.',
            ]
        }
        self.universal_cls_defs = universal_cls_defs

        self.prevad_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Accident': ['An unexpected event, often resulting in damage or injury, involving vehicles, machinery, or other objects.'],
            'AirAccident': universal_cls_defs['AirAccident'],
            'AnimalAttackAnimal': universal_cls_defs['AnimalAttackAnimal'],
            'AnimalAttackHuman': universal_cls_defs['AnimalAttackHuman'],
            'AnimalPredation': universal_cls_defs['AnimalPredation'],
            'CarAccident': universal_cls_defs['CarAccident'],
            'Collapse': universal_cls_defs['Collapse'],
            'CrowdViolence': universal_cls_defs['CrowdViolence'],
            'Explosion': universal_cls_defs['Explosion'],
            'FallDown': universal_cls_defs['FallDown'],
            'FallIntoWater': universal_cls_defs['FallIntoWater'],
            'Fire': universal_cls_defs['Fire'],
            'MechanicalAccident': universal_cls_defs['MechanicalAccident'],
            'ObjectImpact': universal_cls_defs['ObjectImpact'],
            'RangeShooting': [
                'Shooting in a range',
                'A shooting scene in a range, often involving multiple people and firearms.',
                'Firing guns at a shooting range, where people aim at targets from a set distance, often seen bullet holes in targets.',
                'The act of shooting at a target from a distance',
            ],
            'Riot': universal_cls_defs['Riot'],
            'Robbery': universal_cls_defs['Robbery'],
            'Shooting': universal_cls_defs['Shooting'],
            'TrainAccident': universal_cls_defs['TrainAccident'],
            'Violence': universal_cls_defs['Fighting'],
            'WarScene': universal_cls_defs['WarScene'],
        }
        self.ucf_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Abuse': [
                'Intentional beating or abuse of animals like dogs.',
                'Abuse, ill-treatment of pets like dogs or cats.',
                'Beating or kicking pets, torture of animals',
            ],
            'Arrest': ['Police arresting suspects, which may involve pressing them to the ground, controlling hands or aiming with guns.'],
            'Arson': ['The deliberate setting of a fire by someone, usually characterized by flames, smoke, puring gasoline.'],
            'Assault': universal_cls_defs['Assault'],
            'Burglary': ['Burglary, usually characterized by crossing the cashier, breaking doors and windows, and carry things.'],
            'Explosion': universal_cls_defs['Explosion'],
            'Fighting': universal_cls_defs['Fighting'],
            'RoadAccidents': universal_cls_defs['CarAccident'],
            'Robbery': ['Robbing others property through violent means such as beating or holding a gun'],
            'Shooting': universal_cls_defs['Shooting'],
            'Shoplifting': ['Shoplifting, sneak things into bags, clothes or under skirts in stores.'],
            'Stealing': ['Stealing property from cars or stealing motorcycles and batteries.'],
            'Vandalism': ['Damaging vehicles, overturning shelves, or smashing store door.'],
        }
        self.xd_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Fighting': universal_cls_defs['Fighting'],
            'Shooting': universal_cls_defs['Shooting'],
            'Riot': universal_cls_defs['Riot'],
            'Abuse': [
                'Abuse, ill-treatment of people.',
                'Someone suffering from physical abuse',
                'A person is being abused by others.'
            ],
            'Car accident': universal_cls_defs['CarAccident'],
            'Explosion': universal_cls_defs['Explosion'],
        }
        self.msad_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Assault': universal_cls_defs['Assault'],
            'Fighting': universal_cls_defs['Fighting'],
            'People_falling': universal_cls_defs['FallDown'],
            'Robbery': universal_cls_defs['Robbery'],
            'Shooting': universal_cls_defs['Shooting'],
            'Traffic_accident': universal_cls_defs['CarAccident'],
            'Vandalism': [
                'Vandalism',
                'Vandalism, smashing store door, breaking windows.',
            ],
            'Explosion': universal_cls_defs['Explosion'],
            'Fire': universal_cls_defs['Fire'],
            'Object_falling': [
                'Object Falling',
                'Something like trees or buildings collapsed due to strong winds, earthquakes, or impacts.',
            ],
            'Water_incident': [
                'flood scene, with vehicles or furniture submerged',
                'a furious storm',
                'The indoor corridor is filled with water',
            ]
        }
        self.sht_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Cycling': ['Cycling'],
            'Chasing': ['Chasing'],
            'Cart': ['Cart'],
            'Fighting': universal_cls_defs['Fighting'],
            'Skateboarding': ['Skateboarding'],
            'Vehicle': ['The vehicle stopped in the middle of the road'],
            'Running': ['Running'],
            'Jumping': ['Jumping'],
            'Wandering': ['Wandering'],
            'Lifting': ['Lifting'],
            'Robbery': ['Robbery'],
            'Climbing Over': ['Climbing over'],
            'Throwing': ['Throwing', 'Someone accidentally dropped something']
        }
        self.ubnormal_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Running': ['A man is running', 'A woman is running'],
            'Having a Seizure': ['A man is having a seizure', 'A person suddenly started convulsing and then fell to the ground'],
            'Laying Down': universal_cls_defs['FallDown'],
            'Shuffling': ['Shuffling', 'A person walking in a strange and unusual posture'],
            'Walking Drunk': ['Walking Drunk'],
            'People and Car Accident': universal_cls_defs['CarAccident'],
            'Car Crash': universal_cls_defs['CarAccident'],
            'Jumping': ['Jumping', 'A person suddenly jumps', 'A man is jumping'],
            'Fire': universal_cls_defs['Fire'],
            'Smoke': universal_cls_defs['Fire'],
            'Jaywalking': ['Jaywalking', 'A person ran a red light in front of the traffic light', 'A man is jaywalking'],
            'Driving Outside Lane': ['A car is driving outside the lane'],
        }
        self.dota_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'CarAccident': universal_cls_defs['CarAccident'],
        }
        self.nwpu_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Climbing fence': ['A man is climbing fence'],
            'Car crossing square': ['A car crosses square'],
            'Cycling on footpath': ['A man is cycling on the sidewalk'],
            'Kicking trash can': ['A man kicks the trash can'],
            'Jaywalking': ['Jaywalking', 'A person ran a red light in front of the traffic light', 'A man is jaywalking'],
            'Snatching bag': ['A person snatches others bag'],
            'Crossing lawn': ['A person is crossing lawn'],
            'Wrong turn': ['A car is turning in a prohibited area'],
            'Cycling on square': ['A man is cycling on the square'],
            'Chasing': ['A student is chasing another student'],
            'Loitering': ['Loitering'],
            'Scuffle': universal_cls_defs['Fighting'],
            'Littering': ['Littering'],
            'Forgetting backpack': ['Forgetting backpack', 'A man leaves without taking his bag'],
            'U-turn': ['A car makes a U-turn in a prohibited area'],
            'Battering': universal_cls_defs['Assault'],
            'Falling': universal_cls_defs['FallDown'],
            'Driving on wrong side': ['A car is driving on wrong side'],
            'Suddenly stopping cycling in the middle of the road': ['Suddenly stopping cycling in the middle of the road'],
            'Group conflict': universal_cls_defs['CrowdViolence'],
            'Climbing tree': ['A man is climbing a tree'],
            'Stealing': ['Stealing property from others bags.'],
            'Illegal parking': ['Illegal parking'],
            'Trucks': ['Trucks on the wrong place.'],
            'Protest': ['A group of students are protesting'],
            'Playing with water': ['Playing with water'],
            'Photographing in restrict area': ['Photographing in restrict area'],
            'Dogs': ['Dogs on roads'],
        }
        self.ubif_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Fighting': universal_cls_defs['Fighting'],
        }
        self.tad_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Car Accident': universal_cls_defs['CarAccident'],
        }
        self.lad_cls_defs = {
            'Normal': universal_cls_defs['Normal'],
            'Drop': ['A man accidentally drop things'],
            'Loitering': ['Loitering'],
            'Crash': universal_cls_defs['CarAccident'],
            'Violence': universal_cls_defs['Fighting'],
            'FallIntoWater': universal_cls_defs['FallIntoWater'],
            'Fire': universal_cls_defs['Fire'],
            'Fighting': universal_cls_defs['Fighting'],
            'Crowd': universal_cls_defs['CrowdViolence'],
            'Destroy': ['Vandalism'],
            'Falling': universal_cls_defs['FallDown'],
            'Trampled': ['Trampled'],
            'Thiefing': ['Thiefing'],
            'Panic': ['Panic'],
            'Hurt': ['An unexpected event, often resulting in damage or injury, involving vehicles, machinery, or other objects.'],
        }

        # 这是一个特殊的标记，当输入是这个类别时，表示数据没有细类别标注，但是涉及多种类别，此时返回预定义的所有类别
        self._SPECIAL_CLS = 'Abnormal'
        self.cls2text = self.prevad_cls_defs
        self.curr_dataset_name = 'prevad'

    def set_dataset(self, dataset_name: str):
        if dataset_name == self.curr_dataset_name:
            return
        print(f"set verbalizer to `{dataset_name}`")
        self.curr_dataset_name = dataset_name
        if dataset_name == 'prevad':
            self.cls2text = self.prevad_cls_defs
        elif dataset_name == 'ucf':
            self.cls2text = self.ucf_cls_defs
        elif dataset_name == 'xd':
            self.cls2text = self.xd_cls_defs
        elif dataset_name == 'msad':
            self.cls2text = self.msad_cls_defs
        elif dataset_name == 'sht':
            self.cls2text = self.sht_cls_defs
        elif dataset_name == 'ubnormal':
            self.cls2text = self.ubnormal_cls_defs
        elif dataset_name == 'dota':
            self.cls2text = self.dota_cls_defs
        elif dataset_name == 'nwpu':
            self.cls2text = self.nwpu_cls_defs
        elif dataset_name == 'ubif':
            self.cls2text = self.ubif_cls_defs
        elif dataset_name == 'tad':
            self.cls2text = self.tad_cls_defs
        elif dataset_name == 'lad':
            self.cls2text = self.lad_cls_defs
        else:
            raise NotImplementedError


    def __call__(self, inputs: Union[str, List[str]]):
        if type(inputs) is str:
            return random.choice(self.cls2text[inputs])
        elif type(inputs) is list:
            if len(inputs) == 2 and self._SPECIAL_CLS in inputs:
                ret_list = []
                for k, v in self.cls2text.items():
                    ret_list.append(random.choice(v))
                return ret_list
            else:
                return [random.choice(self.cls2text[i]) for i in inputs]

