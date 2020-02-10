import requests

ENGLISH = 'en'


class ParseConceptNet:
    def __init__(self):
        pass

    @staticmethod
    def get_related_words(word):
        word = word.replace(' ', '_')
        obj = requests.get("http://api.conceptnet.io/c/en/{}?limit=100".format(word)).json()

        for edge in obj['edges']:
            if edge['start']['language'] == ENGLISH:
                if 'language' in edge['end'] and edge['end']['language'] == ENGLISH:
                    print("{} {} {}".format(edge['start']['label'], edge['rel']['label'], edge['end']['label']))

    @staticmethod
    def compare_words(word, other_word):
        word = word.replace(' ', '_')
        other_word = other_word.replace(' ', '_')

        obj = requests.get("http://api.conceptnet.io/query?node=/c/en/{}&other=/c/en/{}".format(word, other_word)).json()
        if obj['edges']:
            print("Words are related with weight {}".format(obj['edges'][0]['weight']))
        else:
            print('Words are not related')
