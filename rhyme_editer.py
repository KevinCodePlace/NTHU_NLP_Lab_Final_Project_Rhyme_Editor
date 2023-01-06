import streamlit as st
from annotated_text import annotated_text
import string
import nltk
from nltk import sent_tokenize, word_tokenize
import re

def nltk_install():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_last_word(input_graph):
    input_sents = sent_tokenize(input_graph)
    last_words = []
    for sents in input_sents:
        last_words.append(word_tokenize(re.sub(r'[^\w\s]', '', sents))[-1])
    
    return (last_words,input_sents)

def get_new_paragraph(target_word,last_words,input_sents):
    import  requests
    import json
    from nltk.stem import WordNetLemmatizer
    from transformers import pipeline

    # apply for oxford dictionaries api account
    app_id = 'type_in_your_own'
    app_key = 'type_in_your_own'
    language = 'en-us'
    unmasker = pipeline('fill-mask', model='bert-large-uncased-whole-word-masking')
    
    from nltk.corpus import wordnet
    synonyms_set = set()
    url = 'https://od-api.oxforddictionaries.com/api/v2/entries/'  + language + '/'  + target_word.lower()
    target_r = requests.get(url, headers = {'app_id' : app_id, 'app_key' : app_key})
    target_p = target_r.json()['results'][0]['lexicalEntries'][0]['entries'][0]['pronunciations'][1]['phoneticSpelling']
    new_paragraph = ""

    phome_groups = {
        'vowels_front': {'i', 'ɪ', 'e', 'æ'},
        'vowels_mid': {'ɑ', 'ʌ', 'ə', 'ɚ', 'ɝ'},
        'vowels_back': {'u', 'ʊ', 'ɔ'},
        'diphthongs': {'eɪ', 'o', 'aɪ', 'ɔɪ', 'aʊ'},
        'semi-vowels_liquid': {'w', 'l'},
        'semi-vowels_glides': {'r', 'j'},
        'consonants_stops_voiced': {'b', 'd', 'g'},
        'consonants_stops_unvoiced': {'p', 't', 'k'},
        'nasals': {'m', 'n', 'ŋ'},
        'whisper': {'h'},
        'fricatives_voiced': {'v', 'ð', 'z', 'ʒ'},
        'fricatives_unvoiced': {'f', 'θ', 's', 'ʃ'},
        'affricates': {'tʃ', 'dʒ'}
    }
    
    for i, word in enumerate(last_words[1:]):
        match = 0
        match_last_word = ""
        st.write("---------------------")
        st.write("Checking ", word.lower())
        url = 'https://od-api.oxforddictionaries.com/api/v2/entries/'  + language + '/'  + word.lower()
        r = requests.get(url, headers = {'app_id' : app_id, 'app_key' : app_key})
        source_p = r.json()['results'][0]['lexicalEntries'][0]['entries'][0]['pronunciations'][1]['phoneticSpelling']
        if source_p[-2:] == target_p[-2:]:
            st.write("no need to change word")
            continue
        
        wordnet_synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                wordnet_synonyms.append(lemma.name())
        
        st.write("- looking for synonyms and pronunciations")

        oxford_synonyms = []
        try:
            for oxford_synonym in r.json()['results'][0]['lexicalEntries'][0]['entries'][0]['senses'][0]['synonyms']:
                oxford_synonyms.append(oxford_synonym['text'].lower())
        except:
            st.write("")
                    
        bert_synonyms = []
        for bert_output in unmasker(input_sents[i+1].replace(word, "[MASK]")):
            bert_synonyms.append(bert_output['token_str'])

        synonyms = oxford_synonyms + wordnet_synonyms + bert_synonyms
        synonyms = list(dict.fromkeys(synonyms))
        
        st.write("All the candidate words are: ", synonyms)
        
        for synonym in synonyms:
            if match == 1:
                break
            url = 'https://od-api.oxforddictionaries.com/api/v2/entries/'  + language + '/'  + synonym
            r = requests.get(url, headers = {'app_id' : app_id, 'app_key' : app_key})
            try:
                synonym_p = r.json()['results'][0]['lexicalEntries'][0]['entries'][0]['pronunciations'][1]['phoneticSpelling']
                for last_phome_group in phome_groups:
                    if (synonym_p[-1] in phome_groups[last_phome_group]) and (target_p[-1] in phome_groups[last_phome_group]):
                        st.write('match last group, ', last_phome_group, " ", synonym)
                        match_last_word = synonym
                        if last_phome_group[:5] != 'vowel':
                            for last2_phome_group in phome_groups:
                                if synonym_p[-2] in  phome_groups[last2_phome_group] and target_p[-2] in phome_groups[last2_phome_group]:
                                    st.write("matching target! ", synonym)
                                    match = 1
                                    new_paragraph += input_sents[i+1].replace(word, synonym)
                                    break
            except:
                continue
        if match == 0 and len(match_last_word)!=0:
            new_paragraph += input_sents[i+1].replace(word, match_last_word) + ' '
        elif match == 0:
            new_paragraph += input_sents[i+1] + ' '
    return new_paragraph

def annotated_last_word(old_last_words,new_graph):
    annotated_sents = []
        
    new_sents = sent_tokenize(new_graph)
    new_last_words = []
    for sents in new_sents:
        annotated_sents.append(' '.join(sents.split()[:-1]))
        new_last_words.append(word_tokenize(re.sub(r'[^\w\s]', '', sents))[-1])
        
    for idx,sents in enumerate(annotated_sents):
        annotated_text(sents,(new_last_words[idx],old_last_words[idx]))
    
def text_to_speech(revised_paragraph):
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"  

    from google.cloud import texttospeech

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=revised_paragraph)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')
    
example_paragraph = ["","Speaking on Saturday, England captain Harry Kane said: We are very sad to hear news of him being ill but we wish him well, not just me but the whole England set-up. He's an inspiration, an incredible person. World Cup-winning former Germany forward Jurgen Klinsmann, speaking on BBC One's World Cup coverage on Saturday, said: Pele is just such a wonderful person."]

st.write("# Rhyme Editer")
# add_side_checkbox = st.sidebar.checkbox(
#     "Show original word"
# )
select_paragraph = st.selectbox('Choose the example paragraph:',example_paragraph)
own_paragraph = st.text_area('or type your own!!',select_paragraph)
nltk_install()

paragraph_last_info = get_last_word(own_paragraph)
paragraph_last_words = paragraph_last_info[0]
paragraph_in_sentences = paragraph_last_info[1]

if paragraph_last_words:
    st.write("Below are all of the last words in the paragraph:")
    st.write(paragraph_last_words)
    target_word = st.selectbox('Choose the target rhyme word:',paragraph_last_words)
    agree = st.checkbox(f'Do you agree use "{target_word}" as target word?')
    if agree:
        st.success(f'We now use "{target_word}" as the target word')
        revised_paragraph = get_new_paragraph(target_word,paragraph_last_words,paragraph_in_sentences)
        revised_paragraph = paragraph_in_sentences[0] + ' ' + revised_paragraph
        st.write("Below is the revised result:")
        annotated_last_word(paragraph_last_words,revised_paragraph)
        text_to_speech(revised_paragraph)
        audio_file = open('output.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
        
    else:
        st.error('Please confirm a target word by selecting the checkbox!')

