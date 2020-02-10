import sys
import requests

ENGLISH = 'en'

try:
    word = sys.argv[1].replace(' ', '_')
    other = None
    if len(sys.argv) > 2:
        other = sys.argv[2].replace(' ', '_')
except IOError:
    print('Please pass a word or phrase as a string.')
    sys.exit()

if other:
    obj = requests.get("http://api.conceptnet.io/query?node=/c/en/{}&other=/c/en/{}".format(word, other)).json()
    if obj['edges']:
        print("Words are related with weight {}".format(obj['edges'][0]['weight']))
    else:
        print('Words are not related')
    sys.exit(0)

obj = requests.get("http://api.conceptnet.io/c/en/{}?limit=100".format(word)).json()

for edge in obj['edges']:
    if edge['start']['language'] == ENGLISH:
        if 'language' in edge['end'] and edge['end']['language'] == ENGLISH:
            print("{} {} {}".format(edge['start']['label'], edge['rel']['label'], edge['end']['label']))
