from spacy.tokens import Span
from spacy.matcher import Matcher
import spacy

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe("sentencizer"))


def is_interrogative(tok):
    return tok.dep_ == "advmod" and tok.pos_ == "PRON"


def get_entities(sent):
    # chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        # chunk 2
        # if token is a punctuation mark then move on to the next token(句読点ならスキップ)
        if tok.dep_ != "punct":
            # check: token is a compound word(複合名詞) or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier(修飾語) or not
            if tok.dep_.endswith("mod") == True and not is_interrogative(tok):
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            # chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            # chunk 4
            if tok.dep_.find("obj") == True or is_interrogative(tok):
                ent2 = modifier + " " + prefix + " " + tok.text

            # chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return ent1.strip(), ent2.strip()


def get_relation(sent):

    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]
    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return(span.text)


def sentencize(joke):
    return nlp(joke).sents
