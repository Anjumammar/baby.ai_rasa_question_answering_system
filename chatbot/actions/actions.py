# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util


from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# import numpy as np
from rapidfuzz import process, fuzz


# Loading Model
filename_tokenizer = 'finalized_tokenizer.sav'
loaded_tokenizer = pickle.load(open(filename_tokenizer, 'rb'))
filename2 = 'finalized_model.sav'
loaded_model = pickle.load(open(filename2, 'rb'))


#random Question embedding
filename1="random_question_embeddings.sav"
sentence_embeddings= pickle.load(open(filename1, 'rb'))

#general knowledge Question Answer
filename1="final_embeddings.sav"
sentence_embeddings1= pickle.load(open(filename1, 'rb'))

#boat challenge Question
filename1="boat_challenge.sav"
sentence_embeddings2= pickle.load(open(filename1, 'rb'))

class ActionCalculator(Action):

    def name(self) -> Text:
        return "action_calculator"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # to_calc=tracker.get_latest_entity_values('calc')
        to_calc = next(tracker.get_latest_entity_values('calc'),None)
        # number1 = next(tracker.get_latest_entity_values('number'), None)
        number_list = tracker.get_latest_entity_values('number')
        # number2 = next(tracker.get_latest_entity_values('number2'), None)
        # number1=int(number1)
        # number2=int(number2)
        print(to_calc)
        number_list = list(number_list)
        print(number_list)
        num_list=list(map(int, number_list))
        if not len(number_list):
            dispatcher.utter_message(text="No number found!")
            return []

        if to_calc=='add':
            # print(list(number_list))
            dispatcher.utter_message(text="hereis your answer{}".format(sum(list(map(int, number_list)))))
        elif to_calc=='subtract':

            x=num_list[0] - num_list[1]
            dispatcher.utter_message(text="hereis your answer{}".format(x))
        elif to_calc=='multiply':
            x=num_list[0] * num_list[1]
            dispatcher.utter_message(text="hereis your answer{}".format(x))
        elif to_calc=='devide':
            x=num_list[0] / num_list[1]
            dispatcher.utter_message(text="hereis your answer{}".format(x))



        return []


class generalknowledge(Action):

    def name(self) -> Text:
        return "Action_gk"


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        df = pd.read_csv('Quest_answer16.csv')
        # , encoding = 'latin1'
        # df = pd.read_csv('! line answer foe baby robot.csv', encoding='latin1')

        gk_from_nlu=tracker.latest_message['text']
        latest_question=[]
        latest_question.append(gk_from_nlu)


        corpus = df.Question.tolist()






## for latest Question embedding
        encoded_input1 = loaded_tokenizer(latest_question, padding=True, truncation=True,
                                 return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = loaded_model(**encoded_input1)

        # Perform pooling
        sentence_embeddings_question = generalknowledge.mean_pooling(model_output, encoded_input1['attention_mask'])

        # Normalize embeddings
        sentence_embeddings_question = F.normalize(sentence_embeddings_question, p=2, dim=1)



        cosine_scores = util.cos_sim(sentence_embeddings1, sentence_embeddings_question)
        idx = torch.argmax(cosine_scores).item()
        print(cosine_scores[idx])
        if cosine_scores[idx].item() < 0.6:
            dispatcher.utter_message(text="I don't know that!")
            return []
        print(idx)
        t=corpus[idx]
        print(t)

        y=df.iloc[idx]['Answer']
        print(y)

        dispatcher.utter_message(text="your answer is{}".format(y))


        return []

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)

class Action_random_answer(Action):

    def name(self) -> Text:
        return "Action_random_answer"


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        df = pd.read_csv(r'random_question_baby.csv')
        # , encoding = 'latin1'
        # df = pd.read_csv('! line answer foe baby robot.csv', encoding='latin1')

        gk_from_nlu=tracker.latest_message['text']
        latest_question=[]
        latest_question.append(gk_from_nlu)


        corpus = df.Question.tolist()






## for latest Question embedding
        encoded_input1 = loaded_tokenizer(latest_question, padding=True, truncation=True,
                                 return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = loaded_model(**encoded_input1)

        # Perform pooling
        sentence_embeddings_question = Action_random_answer.mean_pooling(model_output, encoded_input1['attention_mask'])

        # Normalize embeddings
        sentence_embeddings_question = F.normalize(sentence_embeddings_question, p=2, dim=1)



        cosine_scores = util.cos_sim(sentence_embeddings, sentence_embeddings_question)
        idx = torch.argmax(cosine_scores).item()
        if cosine_scores[idx].item() < 0.6:
            dispatcher.utter_message(text="I don't know that!")
            return []
        print(cosine_scores[idx])
        print(idx)
        t=corpus[idx]
        print(t)

        y=df.iloc[idx]['Answer']
        print(y)

        dispatcher.utter_message(text="your answer is{}".format(y))


        return []

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)
class Action_boatChallenge(Action):

    def name(self) -> Text:
        return "Action_robot_challenge"


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        df = pd.read_csv(r'boat_challaenge_5.csv')
        # , encoding = 'latin1'
        # df = pd.read_csv('! line answer foe baby robot.csv', encoding='latin1')

        gk_from_nlu=tracker.latest_message['text']
        latest_question=[]
        latest_question.append(gk_from_nlu)


        corpus = df.Question.tolist()






## for latest Question embedding
        encoded_input1 = loaded_tokenizer(latest_question, padding=True, truncation=True,
                                 return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = loaded_model(**encoded_input1)

        # Perform pooling
        sentence_embeddings_question = Action_random_answer.mean_pooling(model_output, encoded_input1['attention_mask'])

        # Normalize embeddings
        sentence_embeddings_question = F.normalize(sentence_embeddings_question, p=2, dim=1)



        cosine_scores = util.cos_sim(sentence_embeddings2, sentence_embeddings_question)
        idx = torch.argmax(cosine_scores).item()
        if cosine_scores[idx].item() < 0.6:
            dispatcher.utter_message(text="I don't know that!")
            return []
        print(cosine_scores[idx])
        print(idx)
        t=corpus[idx]
        print(t)

        y=df.iloc[idx]['Answer']
        print(y)

        dispatcher.utter_message(text="your answer is{}".format(y))


        return []

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)