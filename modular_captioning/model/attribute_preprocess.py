from scipy import ndimage
from core.utils import *
import csv
import numpy as np
import pandas as pd
import pickle
import os
import json

visual_noun = ['man','people','woman','street','table','person','group','field','tennis','train','room','plate','dog','cat'
       ,'baseball','water','bathroom','sign','kitchen','food','grass','bus','pizza','snow','building','bed','beach'
      ,'ball','men','toilet','city','skateboard','road','clock','player','game','girl','wooden','bear','bench','picture'
      ,'laptop','horse','cake','area','board','giraffe','sink','frisbee','computer','phone','air','truck','desk'
       ,'window','trees','motorcycle','tree','umbrella','park','car','elephant','wall','fire','stop','sky','court'
       ,'kite','skis','child','surfboard','bat','sheep','airplane','boat','bowl','photo','bunch','ocean','bird','couch'
       ,'plane','traffic','zebra','light','hydrant','chair','view', 'background', 'cell','teddy','mirror','fence'
       ,'glass','counter','women','sandwich','giraffes','horses','wave','orange','shirt','tracks','sidewalk','flowers'
       ,'vase','cars','floor','elephants','baby','ski','buildings','trick','children','airport','camera','pole'
       ,'umbrellas','corner','oven', 'track','keyboard','stove','boats','animals','chairs','television','video','soccer'
       ,'box','tv','crowd', 'surf', 'birds', 'door','dogs','plates','lady','runway','banana','decker','motorcycles','body'
       ,'wood','players','guy','skateboarder','cheese','river','night','bedroom', 'house', 'jet', 'coffee', 'bears'
       , 'paper','snowboard', 'meat', 'lights', 'restaurant', 'home', 'skier', 'metal', 'ramp','brick', 'shower', 'racquet'
       , 'remote', 'cup', 'surfer', 'bicycle','passenger', 'items','male','line','hands','face', 'animal', 'intersection'
       ,'book','slice','suitcase', 'mouth', 'zoo','tray', 'enclosure', 'scissors', 'store', 'number', 'donuts', 'batter'
       , 'screen', 'bridge','kids','microwave','carrots', 'bag', 'row', 'tub', 'bottle', 'sand', 'silver','adult','toy'
       , 'boys', 'oranges', 'cabinets', 'girls','furniture','mouse','kid','photograph', 'waves', 'outdoor','seat', 'bread'
       , 'cats','chocolate','fruits','rocks', 'apples','monitor','hair', 'scene', 'meter','drink', 'female', 'salad'
       , 'apple', 'office', 'market', 'shelf', 'fork', 'birthday', 'stone', 'pan', 'watch','snowboarder', 'rain','fries'
       , 'leaves','walls', 'teeth','flower','blanket','dish', 'surfboards', 'sun', 'windows', 'mountains', 'tables', 'bikes'
       , 'police', 'rail', 'bath', 'books','platform','donut', 'uniform', 'country', 'rock','yard','slices','edge', 'helmet'
       , 'cellphone', 'branch','base', 'statue','vases', 'controller','computers','path', 'boards', 'family', 'shore'
       , 'doughnut','case', 'dress', 'sauce', 'tarmac','vehicle','cart', 'town', 'nintendo', 'dinner', 'trains','pizzas'
       ,'surface', 'doughnuts', 'plant', 'toppings','laptops', 'guys', 'hotel','basket','trucks','engine','tricks','lamp'
       ,'appliances', 'rice', 'woods', 'benches', 'passengers','plants', 'gear', 'cattle', 'catcher', 'railroad','flies'
       , 'place', 'beer', 'graffiti', 'match', 'candles', 'curb', 'brush', 'drinks', 'vintage','carriage', 'planes'
       , 'sandwiches', 'toothbrush', 'chicken', 'school', 'bags','bar', 'poles', 'fridge', 'steel', 'shoes', 'dock'
       ,'fireplace', 'beds', 'skateboards','tile','bottles','neck', 'boxes','sofa','trunk', 'pot', 'suitcases', 'pillows'
       , 'bushes', 'equipment','sinks', 'foods','church', 'airplanes','rack','clothes','bicycles', 'vehicles'
       , 'container','pose','poses','potatoes', 'surfers', 'dessert','hay', 'space','feet','subway', 'smiles', 'cream'
       , 'towel','breakfast', 'cement', 'cabinet', 'dishes', 'outdoors', 'trail', 'wire','highway', 'show', 'couches', 'persons'
       ,'games','christmas', 'pool','square', 'military','painting','legs','hillside','backpack', 'blender', 'reflection'
       ,'business', 'controllers', 'spoon', 'arm', 'assortment','clocks','lawn', 'adults', 'soup', 'van', 'wedding'
       , 'shorts', 'team', 'hotdog', 'garden', 'stall','stairs', 'sunglasses', 'rackets','skies', 'bun', 'toddler', 'ledge'
       , 'desktop', 'restroom', 'onions','shelves','gate','forest', 'knife', 'buses', 'lake', 'meal', 'skiers', 'jacket'
       ,'tomatoes','kitten', 'land', 'clouds', 'fish', 'ice', 'flag', 'trash','desert','floors','umpire', 'pillow', 'eyes'
       ,'friends', 'cups','race','rug', 'glove','party', 'steps', 'cakes', 'monitors', 'construction','sunset', 'boarder'
       , 'trailer', 'eggs','foot','electronic', 'doors', 'lunch', 'machine', 'ceiling', 'cage','vegetable','pastries','pots'
       ,'sale', 'pond', 'type','pasta','tan','curtain', 'toilets', 'veggies', 'transit', 'action', 'rider','roof', 'fighter', 'tour'
      , 'towels', 'mans', 'pie','picnic', 'sea','chips', 'suits','pepperoni', 'coat','colors', 'device', 'kinds', 'toys'
      ,'bite', 'signal', 'object','doorway','shade', 'grill','commuter', 'smoke', 'walkway', 'photos', 'scooter', 'net'
      ,'ear','pastry', 'clothing','steam', 'carrot', 'hole','ties','winter', 'leash', 'sides', 'pier', 'houses','leather'
      ,'flight','electric', 'pedestrians','officer', 'vanity', 'peppers', 'containers','cloth','sail', 'streets', 'palm', 'lettuce' 
      ,'tomato','harbor', 'model', 'ship','island','features', 'apartment', 'farm','bow','sheet','pipe','deck','stack', 'garage'
      , 'snowboards', 'barn','ladies','tow', 'propeller', 'papers','sheets', 'lid', 'beans','outfit', 'noodles','mound', 'pants'
      , 'parade', 'pavement','branches', 'bush','trays', 'uniforms', 'log', 'lift','nose', 'ketchup', 'aircraft','bacon'
      ,'meters', 'jets', 'stuff', 'goats', 'airliner', 'lines', 'heads', 'gas','patio', 'part', 'soda', 'ducks', 'platter', 'pans'
      ,'sausage','riders', 'cap','disc','dresser', 'waters','vest','tent','purse', 'mug','carpet', 'utensils', 'color'
      , 'flags', 'frosting', 'costume','foreground', 'sill', 'museum', 'cones', 'mustard', 'greens','roadway', 'shoe', 'mushrooms'
      ,'tea', 'juice', 'built', 'terminal', 'jetliner', 'collection', 'rows','roll','closeup','mother','mother','lamps']

visual_adj = ['young','older','hot','empty','busy','cloudy','lush','sunny','clean','public','bright'
       ,'sandy','wet','cute','dry','flat','commercial','polar','striped','professional','wild','asian','round'
       ,'wooded','assorted','rocky','stainless','broken','messy','closed','wide','urban','antique','fancy','plain'
       ,'blurry','decorative','rainy','ripe','upside','rural','deep','smaller','fake','fried','candle','residential']

non_visual = ['parked','sits','covered','open','filled','stands','stand','holds','topped','hit','set'
             ,'walk','rides','shown','dressed','cut','made','half','play','surrounded', 'colored','decorated'
             ,'lined','walks','past','displayed','nice'
             ,'attached', 'cross','lit','painted','catch','plastic','beautiful','stopped','clear','ride','eat','time'
             ,'eaten','shot','pitch','perched','types','tiled','modern','variety','jump','gathered','watches','mounted'
             ,'fenced','crowded','concrete','sliced','distance','end','pulled','served','takes','shaped','plays','control'
             ,'control','work','run','serve','prepares','style','cooked','stacked','swings','cluttered', 'docked'
             ,'throw','lays','drives','drawn','french','jumps','eats','even','fashioned','seated'
             ,'someones','turn','tied','arranged','kind','pointing','overhead','loaded','reads','wrapped'
             , 'make','baked','prepared', 'hold','rest','partially','held','opened','sized','produce','brightly','includes'
             ,'appears','hard','waits','happy','smart','personal','enclosed','multi','pick','fun','makes'
             ,'turned','semi','rests','throws','sort','laid','cover','ornate','power','beige','performs']


position = ['top','front','side','close','inside','back','high','underneath','left','center','mid','nearby','atop','low','alongside'
           ,'beneath', 'bottom']

counting = ['one','two','three','four','five','couple','lot','pair','lots','pile','pieces','single','multiple','lone','flock','bunches',]

action = ['sitting','standing','holding','riding','playing','walking','flying','laying','wearing','eating','living'
         ,'parking','jumping','swinging','talking','watching','posing','traveling','smiling','carrying','waiting','running'
         ,'preparing','surfing','lying','pulling','sleeping','showing','swing','throwing','crossing','making','drinking'
         ,'coming','hitting','leaning','dining','resting','working','setting','moving','skateboarding','passing','brushing'
         ,'cooking','enjoying','performing','reading','catching','flowing','reaching','putting','serving','staring','including'
         ,'sticking','boarding','swimming','snowboarding','landing','skating','facing','kneeling','overlooking','petting','giving'
         ,'racing','leading','railing','shopping','displaying','sailing','sliding','loading','touching', 'writing','kicking'
         ,'relaxing','approaching','attempting','growing','practicing']

color = ['white','black','red','blue','green','brown','yellow','orange','colorful','gray','dark','purple','grey','gold']
size = ['large','small','big','tall','huge','giant','tiny']
people_list=['person','man','woman','boy','girl','child','people','men','women','boys','girls','player','lady','children',
             'players','guy','skateboarder','surfer','passenger','male','kids','adults','adult','kid','female','police',
             'guys','pitcher','catcher','mother','officer','ladies','spectators','skateboarders','workers','snowboarders',
             'pedestrian']

save_names = ['color_labels.pkl','count_labels.pkl','size_labels.pkl'
              ,'semantic_labels.pkl','spatial_labels.pkl']
word_bags = [color,counting,size,action,position]
data = load_pickle('./data/train/train.annotations.pkl')

for i in range(5):
	temp = word_bags[i]
	labels_onehot = np.zeros([data.shape[0],len(temp)])
	print(labels_onehot.shape)
	for i, caption in enumerate(data['caption']):
	    source_list = caption.split(' ')
	    for j in range(len(temp)):
	        if temp[j] in source_list:
	            labels_onehot[i,j] = 1
	print(labels_onehot.shape)
	save_pickle(labels_onehot, './data/train/'+save_names[i])


