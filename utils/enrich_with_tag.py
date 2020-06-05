# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nltk
look_up_file = 'look_up.txt'
TAG_SEP = " "
CAND_SEP = "|"
SENTENCE_TAG_DICT = None

def read_all_lines(path):
    all_lines = None
    with open(path) as f:
        all_lines = f.readlines()
    return all_lines

def make_dict(path):
    all_lines = read_all_lines(path)
    lines_tag = {}

    for line in all_lines:
        line_tag_arr = line.split(",")
        sentence =  line_tag_arr[0]
        tag = line_tag_arr[1].rstrip()

        lines_tag[sentence] = tag
        lines_tag[sentence.rstrip()] = tag

    return lines_tag

def get_dict():
    global SENTENCE_TAG_DICT
    if SENTENCE_TAG_DICT is None:
        SENTENCE_TAG_DICT = make_dict(look_up_file)
    return SENTENCE_TAG_DICT


######
def enrich_sentences(sentences):
    tagged_sentence = []
    sentence_tag_dict = get_dict()
    for sentence in sentences:
        tag = sentence_tag_dict[sentence]
        tagged = tag+TAG_SEP+sentence
        tagged_sentence.append(tagged)
    return tagged_sentence
    

def enrich_utterence(utterence):
    utterence_sentences = nltk.sent_tokenize(utterence)
    tagged_sentences = enrich_sentences(utterence_sentences)
    enrich_utterence_combined = ""
    for sentence in tagged_sentences:
        enrich_utterence_combined = enrich_utterence_combined+" "+sentence
    enrich_utterence_combined = enrich_utterence_combined.strip()
    return enrich_utterence_combined

def enrich_text(text):
    utterece_split = text.split(CAND_SEP)
    enrich_text_arr = [] 
    final_text_block = ""
    for utterence in utterece_split:
        tagged_utterence = enrich_utterence(utterence)
        enrich_text_arr.append(tagged_utterence)
        final_text_block = final_text_block+tagged_utterence+CAND_SEP
    
    
    final_text_block = final_text_block[0:len(final_text_block)-len(CAND_SEP)]
    return final_text_block


#sentence = "I get nostalgic all the time."
#utterence = "Thanks_comma_ I'm a little scared but it will be worth it. Being from the UK bad teeth are in my genes."
#ans =  enrich_utterence(utterence)
#init_dict(look_up_file)
text_block ="oh that is scary.  I hope you were ok.  make sure your doors are locked|Do you know them we'll enough to ask them to stop?|Aw Im sorry. Are you doing better now?|that's great_comma_ i'm glad she's nice to you.|That is cool! I love flash backs from the past especially when it comes to television. I sometimes watch old shows on youtube that I used to watch as a kid. Love it!|Yup gotta lay down the law|I'm sure they will be great to him and he'll bring them joy as well!|Thats never good when you have a bad manager.  What made them so bad?|Yea I hear ya. Im stuck here at work for another day too.|What was her name? Have you read Rainbow Bridge?|That is awesome. I never even get a second date A husband may not be in the cards lol|Sounds like something on a movie! What a work day right_comma_ my god. Bring extras next time? lol |Don't forget a first aid kit! We always need those on camping trips!|Whatever it is_comma_ I hope you have a good time!|Sounds like a great friend! Do they just listen or offer helpful advice?|Ooo! That's exciting! Congrats on your baby! I hope your guests enjoy their time. |That sounds pretty disturbing.  Have you called the cops?|That is just gross!  Have you spoken to her about it?|That is good to hear. What happened that made you think so?|That is great_comma_ I believe that you will do amazing! |Hahah that is funny. Were your parents angry?|Ah okay_comma_ nice for the kiddos then. They grow up fast_comma_ so hold onto these moments while you can:)|well that sounds fun!|Why would you do that?|That must be hard to do_comma_ he wants to be beside you. He is going to miss you.|Hey_comma_ why not? It's unexpected money!|Well that is a good way of doing things as long are you getting the job right in my opinion.|That sounds scary! I always avoid going to the dentist until I absolutely have to. I know that's pretty bad!|Oh_comma_ I see. I think I kind of understand. It at least sounds like you're a really self-aware person_comma_ and appreciative of the small things_comma_ and that you have decent perspective on things. The emotions might be difficult to deal with_comma_ but at least there are some positives there for you to take stock of|Oh_comma_ that's so sad. What earthquake?|You sound very confident! I wish I had your confidence when I was in school.|At least you're not getting sued|yes. I wish I could send you a clip of it from the concert. It was awesome. Maybe you could catch def leppard next time. |Right!? It's weird_comma_ you buy them all the toys in the world_comma_ but they still go after your stuff.|Aw man_comma_ I'm sorry to hear that. Is there anything I can do to make you feel better? |I would imagine_comma_ that is a lot to drink!|I would have been disappointed_comma_ too. |I bet it wasn't Comcast! They wouldn't even give you a cup of water for free.|I can imagine that you did want to run and hide. Sometimes husbands get us into all kinds of situations.|That's awesome!! I love jeeps!  congratulations_comma_ that's great news!|Oh dear lord.. Was he sick or something?|Music can be so touching to the soul.|That is absolutely awful! How long did it take for your ankle to heal?|That should be fun. Any big planes?|That's too bad. I hope you had a blast!|thats better_comma_ but sorry for the disappointment she had caused|Hard work pays off in the long-run. If your work is truly authentic in its nature_comma_ it is definitely going to be recognized and appreciated|So i am guessing she gave you a good surprise?|Oh my goodness_comma_ what was the terrible news?|What did they do to you?|I get that.  It's weird when I forget the name of someone I knew for years.|Oh_comma_ I shouldn't be ungrateful_comma_ lol! At least he kills them before delivering them. A friends cat used to deliver live mice into the bathtub!|I'm sorry. That sounds bad. Flying always makes me nervous but I usually make it ok.|Wow.. that's unbelievable. I wonder how his cholesterol is. |Its goof to have support when it does change.|Oh my gosh that's insane. Well_comma_ I'm sure you'll be able to control your impulses in the future. |You must really trust your friend! How did that turn out?|I like to do both as well. TV if I'm feeling lazy_comma_ reading if I feel more ambitious. |That's amazing! Tell him congratulations from a friendly stranger.|Oh no.. I'm sure you did not feel great when you found out|She is very nice person. That is great.|I think they have some leftover fireworks that they don't want to sit on. They should just use them next time.|Oh good.  Hopefully she'll try to fix it with physical therapy first.  Some doctors love to do surgery right away when its not necessary.  Recommend that she get a second opinion.  Hope she resolves it!|That's bad news. I hope that everyone is healthy and doing well.|I think that's a kind of natural reaction_comma_ especially if it is just a hobby. I would have a fear of being judged.|That's really cool of them. You weren't expecting it at all_comma_ hu?|But it is also the best feeling ever! Enjoy every moment!|must be nice to have someone that loves you|That is really scary. Are you home alone?|It's good to get to spend time with those you love. I am happy for you.|That story could become a movie script.|Are you going to do anything about it in the future?|Go hand out with friends or host a night over sleep_comma_that would be fun.|Oh my god_comma_ that sounds like the most terrifying thing ever! Where was this so I know never to go there?|They have live traps to catch and release that are only a couple bucks. Mice really are sweet natured.|Did you tell someone about it so they could take care of it?|Everyone gets jealous_comma_ It when you act on unfounded stuff that is the issue.|You are right about that. Owning a car can get expensive.|Its horrible to have to got through things like thaty|It depends on the machine you are working on.|Oh_comma_ I see. A workable shelf then. |I don't know whether that's a good or bad thing. Which is it?|It will be ok in the end.|Is he healthy?|I am glad no one in your family is sick.|I agree_comma_ do you have a favorite artist?|how sweet and thoughtful of you|Confidence is super important|Kinda makes you think even more since its such a good deal that people are just like get rid of it - deff look into it! |Yeah_comma_ I'm glad you didn't do anything you would regret. I hope he is better for you in the future.|Maybe she does not have the hots for you anymore.|wow_comma_ so what are you going to do with it?|Yes_comma_ we can always count on Him to know the bigger picture and have faith that He's working for our good.  Hope you have a blessed day!|I love Colorado_comma_ my favorite state.|That is a lot of walking. I can hardly walk 1 mile.|Get her flowers and a gift certificate to a nice massage place!|The frustration they go through well at least we don't have to take the kids or parents home.|It'll be okay I promise.  First few days will be weird but you'll slip in no problem once you adjust to your schedule.|Well_comma_ At least you're in one piece. you could have been in several.|Very nice.  How many shoes do you have?"
ans = enrich_text(text_block)
z = 1







