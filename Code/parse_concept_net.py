import requests

ENGLISH = 'en'
DIRECT_RELATION = 'RelatedTo'


class ParseConceptNet:
    def __init__(self, print_results=True):
        self.print_results = print_results

    # Returns related words, sorted by weight
    # If only_direct_relations is specified, only words <other> such that <word> RelatedTo <other> will be shown
    def get_related_words(self, word, only_direct_relations=False):
        word = word.replace(' ', '_')
        obj = requests.get("http://api.conceptnet.io/c/en/{}?limit=100".format(word)).json()
        relations = []

        for edge in obj['edges']:
            if edge['start']['language'] == ENGLISH:
                if 'language' in edge['end'] and edge['end']['language'] == ENGLISH:
                    relation_start = edge['start']['label']
                    relation_type = edge['rel']['label']
                    relation_end = edge['end']['label']
                    if not only_direct_relations or relation_start == word and relation_type == DIRECT_RELATION:
                        relations.append(relation_end)
                        if self.print_results:
                            print("{} {} {}".format(relation_start, relation_type, relation_end))
        return relations

    # Returns weight between two words
    def compare_words(self, word, other_word):
        word = word.replace(' ', '_')
        other_word = other_word.replace(' ', '_')

        obj = requests.get("http://api.conceptnet.io/query?node=/c/en/{}&other=/c/en/{}".format(word, other_word)).json()
        if obj['edges']:
            weight = obj['edges'][0]['weight']
            if self.print_results:
                print('Words are related with weight', weight)
            return weight
        else:
            print('Words are not related')
            return 0
